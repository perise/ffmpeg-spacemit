/*
 * spacemitmppdec.c — SpacemiT K1 MPP hardware video decoder (FFmpeg codec plugin)
 *
 * Uses the SpacemiT MPP library (libspacemit_mpp) for H.264/HEVC/VP8/VP9
 * hardware decoding on the K1 SoC's Linlon V5/V7 VPU.
 *
 * Architecture matches the working vdec_demo.c exactly:
 *   - Send thread: feeds packets via VDEC_Decode (like demo's send_thread_fn)
 *   - Receive thread: calls VDEC_RequestOutputFrame in tight loop (like demo's main)
 *   - FFmpeg thread: pulls decoded frames from a queue
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

#include "avcodec.h"
#include "bsf.h"
#include "codec_internal.h"
#include "decode.h"
#include "libavutil/log.h"
#include "libavutil/mem.h"
#include "libavutil/pixfmt.h"

#include "vdec.h"
#include "module.h"
#include "packet.h"
#include "frame.h"
#include "para.h"

#define INPUT_BUF_SIZE  (1024 * 1024)
#define SEND_RING_SIZE  64
#define FRAME_RING_SIZE 8

typedef struct DecodedFrame {
    uint8_t *y_data;
    uint8_t *uv_data;
    int      y_linesize;
    int      uv_linesize;
    int      width;
    int      height;
    int64_t  pts;
    int      valid;
} DecodedFrame;

typedef struct SpacemiTMPPDecContext {
    const AVClass  *class;
    MppVdecCtx     *mpp_ctx;
    MppPacket      *mpp_pkt;
    MppFrame       *mpp_frame;
    AVBSFContext   *bsf;

    int             width;
    int             height;
    int             draining;
    int             initialized;

    /* Send thread — feeds packets to MPP */
    pthread_t       send_thread;
    int             send_thread_running;
    atomic_int      send_thread_stop;

    /* SPSC ring: FFmpeg thread → send thread */
    uint8_t        *send_data[SEND_RING_SIZE];
    int             send_size[SEND_RING_SIZE];
    int64_t         send_pts[SEND_RING_SIZE];
    atomic_int      send_head;
    atomic_int      send_tail;
    atomic_int      send_eos;

    /* Receive thread — retrieves decoded frames from MPP */
    pthread_t       recv_thread;
    int             recv_thread_running;
    atomic_int      recv_thread_stop;
    atomic_int      recv_resolution_changed;

    /* SPSC ring: receive thread → FFmpeg thread */
    DecodedFrame    frame_ring[FRAME_RING_SIZE];
    atomic_int      frame_head;
    atomic_int      frame_tail;

    /* Shared state from receive thread */
    atomic_int      out_width;
    atomic_int      out_height;
    atomic_int      eos_received;
} SpacemiTMPPDecContext;

static MppCodingType codec_id_to_mpp(enum AVCodecID id)
{
    switch (id) {
    /* Note: the libspacemit_mpp wrapper has its own whitelist in
     * checkInputParameters() that is more restrictive than the underlying
     * Linlon V5/V7 V4L2 driver's advertised format list. Currently it only
     * accepts H264, H265, MJPEG, VP8, VP9, MPEG2, MPEG4. JPEG (coding 7) is
     * rejected even though MJPEG (coding 6) is accepted, so map both
     * AV_CODEC_ID_MJPEG and AV_CODEC_ID_MJPEGB to CODING_MJPEG.
     * H263, VC1, CAVS, AVS2 are advertised by the V4L2 driver but rejected
     * by the wrapper — they are not exposed as FFmpeg decoders here. */
    case AV_CODEC_ID_H264:        return CODING_H264;
    case AV_CODEC_ID_HEVC:        return CODING_H265;
    case AV_CODEC_ID_VP8:         return CODING_VP8;
    case AV_CODEC_ID_MPEG2VIDEO:  return CODING_MPEG2;
    case AV_CODEC_ID_MPEG4:       return CODING_MPEG4;
    default:                      return CODING_UNKNOWN;
    }
}

/* ---- Send thread (matches demo's send_thread_fn) ---- */

static int send_ring_enqueue(SpacemiTMPPDecContext *s, uint8_t *data, int size, int64_t pts)
{
    int head = atomic_load_explicit(&s->send_head, memory_order_relaxed);
    int next = (head + 1) % SEND_RING_SIZE;
    int tail = atomic_load_explicit(&s->send_tail, memory_order_acquire);
    if (next == tail) return -1;
    s->send_data[head] = data;
    s->send_size[head] = size;
    s->send_pts[head]  = pts;
    atomic_store_explicit(&s->send_head, next, memory_order_release);
    return 0;
}

static void *send_thread_func(void *arg)
{
    SpacemiTMPPDecContext *s = arg;

    while (!atomic_load_explicit(&s->send_thread_stop, memory_order_relaxed)) {
        int tail = atomic_load_explicit(&s->send_tail, memory_order_relaxed);
        int head = atomic_load_explicit(&s->send_head, memory_order_acquire);

        if (tail == head) {
            if (atomic_load_explicit(&s->send_eos, memory_order_acquire)) {
                PACKET_SetLength(s->mpp_pkt, 0);
                PACKET_SetEos(s->mpp_pkt, MPP_TRUE);
                for (int i = 0; i < 100; i++) {
                    S32 r = VDEC_Decode(s->mpp_ctx, PACKET_GetBaseData(s->mpp_pkt));
                    if (r == MPP_OK) break;
                    usleep(5000);
                }
                return NULL;
            }
            usleep(1000);
            continue;
        }

        uint8_t *data = s->send_data[tail];
        int      size = s->send_size[tail];
        int64_t  pts  = s->send_pts[tail];

        if (size > 0 && size <= INPUT_BUF_SIZE) {
            void *buf = PACKET_GetDataPointer(s->mpp_pkt);
            memcpy(buf, data, size);
            PACKET_SetLength(s->mpp_pkt, size);
            PACKET_SetPts(s->mpp_pkt, pts);
            PACKET_SetEos(s->mpp_pkt, MPP_FALSE);

            static int sent_count = 0;
            for (int retry = 0; retry < 5000; retry++) {
                if (atomic_load_explicit(&s->send_thread_stop, memory_order_relaxed))
                    return NULL;
                S32 r = VDEC_Decode(s->mpp_ctx, PACKET_GetBaseData(s->mpp_pkt));
                if (r == MPP_OK) {
                    sent_count++;
                    break;
                }
                if (r == MPP_DATAQUEUE_FULL || r == MPP_POLL_FAILED ||
                    r == MPP_CODER_NO_DATA) {
                    /* Transient back-pressure from the V4L2 input queue
                     * waiting for output drain; just retry. */
                    usleep(2000);
                    continue;
                }
                av_log(NULL, AV_LOG_ERROR,
                       "spacemit_mpp: VDEC_Decode returned %d\n", r);
                break;
            }
        }

        av_free(data);
        atomic_store_explicit(&s->send_tail, (tail + 1) % SEND_RING_SIZE,
                              memory_order_release);
    }
    return NULL;
}

/* ---- Receive thread (matches demo's main loop) ---- */

static int frame_ring_enqueue(SpacemiTMPPDecContext *s, DecodedFrame *df)
{
    int head = atomic_load_explicit(&s->frame_head, memory_order_relaxed);
    int next = (head + 1) % FRAME_RING_SIZE;
    int tail = atomic_load_explicit(&s->frame_tail, memory_order_acquire);
    if (next == tail) return -1;
    s->frame_ring[head] = *df;
    atomic_store_explicit(&s->frame_head, next, memory_order_release);
    return 0;
}

static void *recv_thread_func(void *arg)
{
    SpacemiTMPPDecContext *s = arg;

    while (!atomic_load_explicit(&s->recv_thread_stop, memory_order_relaxed)) {
        int ret = VDEC_RequestOutputFrame(s->mpp_ctx, FRAME_GetBaseData(s->mpp_frame));

        if (ret == MPP_OK) {
            if (FRAME_GetEos(s->mpp_frame) != FRAME_NO_EOS) {
                VDEC_ReturnOutputFrame(s->mpp_ctx, FRAME_GetBaseData(s->mpp_frame));
                atomic_store(&s->eos_received, 1);
                return NULL;
            }

            MppVdecPara *p = &s->mpp_ctx->stVdecPara;
            int w = p->nWidth;
            int h = p->nHeight;
            int stride = p->nStride > 0 ? p->nStride : w;

            if (w > 0 && h > 0) {
                uint8_t *y_src  = (uint8_t *)FRAME_GetDataPointer(s->mpp_frame, 0);
                uint8_t *uv_src = (uint8_t *)FRAME_GetDataPointer(s->mpp_frame, 1);

                if (y_src && uv_src) {
                    /* Copy frame data to our own buffer */
                    size_t y_size = (size_t)h * w;
                    size_t uv_size = (size_t)(h / 2) * w;
                    uint8_t *y_buf = av_malloc(y_size);
                    uint8_t *uv_buf = av_malloc(uv_size);

                    if (y_buf && uv_buf) {
                        for (int i = 0; i < h; i++)
                            memcpy(y_buf + i * w, y_src + i * stride, w);
                        for (int i = 0; i < h / 2; i++)
                            memcpy(uv_buf + i * w, uv_src + i * stride, w);

                        /* The K1 Linlon V5/V7 plugin does not propagate
                         * the input packet PTS through to the decoded frame
                         * (FRAME_GetPts always returns the same bogus value
                         * of 2^32). Returning that as the AVFrame pts causes
                         * FFmpeg's muxer to drop everything after the first
                         * frame as non-monotonic. AV_NOPTS_VALUE tells FFmpeg
                         * to derive PTS from the stream framerate. */
                        DecodedFrame df = {
                            .y_data = y_buf, .uv_data = uv_buf,
                            .y_linesize = w, .uv_linesize = w,
                            .width = w, .height = h,
                            .pts = AV_NOPTS_VALUE,
                            .valid = 1,
                        };

                        /* Block until the consumer ring has space.
                         * Dropping frames here loses real video data and
                         * makes the decoder emit only a handful of frames. */
                        while (!atomic_load_explicit(&s->recv_thread_stop,
                                                     memory_order_relaxed) &&
                               frame_ring_enqueue(s, &df) < 0) {
                            usleep(1000);
                        }
                        if (atomic_load_explicit(&s->recv_thread_stop,
                                                 memory_order_relaxed)) {
                            av_free(y_buf);
                            av_free(uv_buf);
                        } else {
                            atomic_store(&s->out_width, w);
                            atomic_store(&s->out_height, h);
                        }
                    } else {
                        av_free(y_buf);
                        av_free(uv_buf);
                    }
                }
            }
            VDEC_ReturnOutputFrame(s->mpp_ctx, FRAME_GetBaseData(s->mpp_frame));

        } else if (ret == MPP_RESOLUTION_CHANGED) {
            /* Like demo: just continue polling */
            usleep(2000);
        } else if (ret == MPP_CODER_EOS) {
            atomic_store(&s->eos_received, 1);
            return NULL;
        } else {
            usleep(2000);
        }
    }
    return NULL;
}

/* ---- FFmpeg interface ---- */

static int spacemit_decode_init(AVCodecContext *avctx)
{
    SpacemiTMPPDecContext *s = avctx->priv_data;
    int ret;

    s->width  = avctx->width;
    s->height = avctx->height;

    /* Bitstream filter: convert MP4-style length-prefixed NALs to Annex-B
     * if needed. Both filters are no-ops on already-Annex-B input, so it's
     * safe to always wire them up. NB: av_bsf_alloc already allocates
     * par_in/par_out -- overwriting par_in leaks memory, and aliasing
     * extradata via direct pointer assignment causes a double-free at close
     * (the BSF frees par_in->extradata, then avctx frees its own copy).
     * avcodec_parameters_from_context() does the proper deep copy. */
    {
        const char *bsf_name = NULL;
        if      (avctx->codec_id == AV_CODEC_ID_H264) bsf_name = "h264_mp4toannexb";
        else if (avctx->codec_id == AV_CODEC_ID_HEVC) bsf_name = "hevc_mp4toannexb";
        if (bsf_name) {
            const AVBitStreamFilter *f = av_bsf_get_by_name(bsf_name);
            if (f) {
                ret = av_bsf_alloc(f, &s->bsf);
                if (ret < 0) return ret;
                ret = avcodec_parameters_from_context(s->bsf->par_in, avctx);
                if (ret < 0) { av_bsf_free(&s->bsf); return ret; }
                s->bsf->time_base_in = avctx->time_base;
                ret = av_bsf_init(s->bsf);
                if (ret < 0) { av_bsf_free(&s->bsf); return ret; }
            }
        }
    }

    s->mpp_ctx = VDEC_CreateChannel();
    if (!s->mpp_ctx) return AVERROR_EXTERNAL;

    s->mpp_ctx->eCodecType = CODEC_V4L2_LINLONV5V7;
    s->mpp_ctx->pModule    = module_init(CODEC_V4L2_LINLONV5V7);
    if (!s->mpp_ctx->pModule) return AVERROR_EXTERNAL;

    /* Initial size is just a hint for V4L2 input/output buffer allocation;
     * the decoder reconfigures on V4L2_EVENT_SOURCE_CHANGE once it parses
     * the real SPS. Use the size from avctx if the demuxer/parser provided
     * one (covers MP4/MKV/etc.), and only fall back to 1080p when we have
     * absolutely no information (e.g. raw elementary streams). */
    s->mpp_ctx->stVdecPara.eCodingType             = codec_id_to_mpp(avctx->codec_id);
    s->mpp_ctx->stVdecPara.nWidth                  = avctx->width  > 0 ? avctx->width  : 1920;
    s->mpp_ctx->stVdecPara.nHeight                 = avctx->height > 0 ? avctx->height : 1080;
    s->mpp_ctx->stVdecPara.eOutputPixelFormat      = PIXEL_FORMAT_NV12;
    s->mpp_ctx->stVdecPara.nHorizonScaleDownRatio  = 1;
    s->mpp_ctx->stVdecPara.nVerticalScaleDownRatio = 1;
    s->mpp_ctx->stVdecPara.bIsFrameReordering      = MPP_TRUE;
    s->mpp_ctx->stVdecPara.bThumbnailMode          = MPP_FALSE;
    s->mpp_ctx->stVdecPara.bNoBFrames              = MPP_FALSE;

    ret = VDEC_Init(s->mpp_ctx);
    if (ret != MPP_OK) return AVERROR_EXTERNAL;

    s->mpp_pkt = PACKET_Create();
    if (!s->mpp_pkt) return AVERROR_EXTERNAL;
    PACKET_Alloc(s->mpp_pkt, INPUT_BUF_SIZE);

    s->mpp_frame = FRAME_Create();
    if (!s->mpp_frame) return AVERROR_EXTERNAL;

    atomic_init(&s->send_head, 0);
    atomic_init(&s->send_tail, 0);
    atomic_init(&s->send_eos, 0);
    atomic_init(&s->send_thread_stop, 0);
    atomic_init(&s->recv_thread_stop, 0);
    atomic_init(&s->recv_resolution_changed, 0);
    atomic_init(&s->frame_head, 0);
    atomic_init(&s->frame_tail, 0);
    atomic_init(&s->out_width, 0);
    atomic_init(&s->out_height, 0);
    atomic_init(&s->eos_received, 0);

    /* Start receive thread first (like demo: main thread starts before send) */
    ret = pthread_create(&s->recv_thread, NULL, recv_thread_func, s);
    if (ret != 0) return AVERROR_EXTERNAL;
    s->recv_thread_running = 1;

    /* Start send thread */
    ret = pthread_create(&s->send_thread, NULL, send_thread_func, s);
    if (ret != 0) return AVERROR_EXTERNAL;
    s->send_thread_running = 1;

    s->initialized = 1;
    av_log(avctx, AV_LOG_INFO, "SpacemiT MPP decoder initialized: %s %dx%d\n",
           avcodec_get_name(avctx->codec_id), avctx->width, avctx->height);
    return 0;
}

static int spacemit_decode_receive_frame(AVCodecContext *avctx, AVFrame *frame)
{
    SpacemiTMPPDecContext *s = avctx->priv_data;
    int ret;

    /* Try to get a decoded frame from the receive thread's queue */
    int tail = atomic_load_explicit(&s->frame_tail, memory_order_relaxed);
    int head = atomic_load_explicit(&s->frame_head, memory_order_acquire);

    if (tail != head) {
        DecodedFrame *df = &s->frame_ring[tail];

        int w = atomic_load(&s->out_width);
        int h = atomic_load(&s->out_height);
        if (w != s->width || h != s->height) {
            av_log(avctx, AV_LOG_INFO, "Resolution: %dx%d -> %dx%d\n",
                   s->width, s->height, w, h);
            s->width  = w;
            s->height = h;
        }
        avctx->width  = w;
        avctx->height = h;

        frame->format = AV_PIX_FMT_NV12;
        frame->width  = df->width;
        frame->height = df->height;
        frame->pts    = df->pts;

        ret = av_frame_get_buffer(frame, 0);
        if (ret < 0) {
            av_free(df->y_data);
            av_free(df->uv_data);
            atomic_store_explicit(&s->frame_tail, (tail + 1) % FRAME_RING_SIZE,
                                  memory_order_release);
            return ret;
        }

        for (int i = 0; i < df->height; i++)
            memcpy(frame->data[0] + i * frame->linesize[0],
                   df->y_data + i * df->y_linesize, df->width);
        for (int i = 0; i < df->height / 2; i++)
            memcpy(frame->data[1] + i * frame->linesize[1],
                   df->uv_data + i * df->uv_linesize, df->width);

        av_free(df->y_data);
        av_free(df->uv_data);
        df->y_data = NULL;
        df->uv_data = NULL;

        atomic_store_explicit(&s->frame_tail, (tail + 1) % FRAME_RING_SIZE,
                              memory_order_release);
        return 0;
    }

    /* No frame available */
    if (atomic_load(&s->eos_received))
        return AVERROR_EOF;

    if (s->draining) {
        /* Wait for remaining frames or EOS */
        for (int i = 0; i < 500; i++) {
            usleep(2000);
            head = atomic_load_explicit(&s->frame_head, memory_order_acquire);
            if (head != atomic_load_explicit(&s->frame_tail, memory_order_relaxed))
                return spacemit_decode_receive_frame(avctx, frame);
            if (atomic_load(&s->eos_received))
                return AVERROR_EOF;
        }
        return AVERROR_EOF;
    }

    /* Get packets from FFmpeg and enqueue for send thread */
    for (int batch = 0; batch < 8; batch++) {
        AVPacket *pkt = av_packet_alloc();
        if (!pkt) return AVERROR(ENOMEM);

        ret = ff_decode_get_packet(avctx, pkt);
        if (ret == AVERROR_EOF) {
            av_packet_free(&pkt);
            av_log(avctx, AV_LOG_INFO, "All packets sent, entering drain mode\n");
            atomic_store_explicit(&s->send_eos, 1, memory_order_release);
            s->draining = 1;
            /* Wait a bit for remaining frames */
            for (int i = 0; i < 1000; i++) {
                usleep(2000);
                head = atomic_load_explicit(&s->frame_head, memory_order_acquire);
                if (head != atomic_load_explicit(&s->frame_tail, memory_order_relaxed))
                    return spacemit_decode_receive_frame(avctx, frame);
                if (atomic_load(&s->eos_received))
                    return AVERROR_EOF;
            }
            return AVERROR_EOF;
        } else if (ret < 0) {
            av_packet_free(&pkt);
            return ret;
        }

        if (s->bsf) {
            ret = av_bsf_send_packet(s->bsf, pkt);
            if (ret < 0) { av_packet_free(&pkt); return ret; }
            ret = av_bsf_receive_packet(s->bsf, pkt);
            if (ret < 0) { av_packet_free(&pkt); return AVERROR(EAGAIN); }
        }

        if (pkt->size > 0) {
            int sz = pkt->size > INPUT_BUF_SIZE ? INPUT_BUF_SIZE : pkt->size;
            uint8_t *data = av_malloc(sz);
            if (!data) { av_packet_free(&pkt); return AVERROR(ENOMEM); }
            memcpy(data, pkt->data, sz);
            if (send_ring_enqueue(s, data, sz, pkt->pts) < 0)
                av_free(data);
        }
        av_packet_free(&pkt);

        /* Check if a frame arrived */
        head = atomic_load_explicit(&s->frame_head, memory_order_acquire);
        if (head != atomic_load_explicit(&s->frame_tail, memory_order_relaxed))
            return spacemit_decode_receive_frame(avctx, frame);
    }

    return AVERROR(EAGAIN);
}

static void stop_threads(SpacemiTMPPDecContext *s)
{
    if (s->send_thread_running) {
        atomic_store_explicit(&s->send_thread_stop, 1, memory_order_release);
        pthread_join(s->send_thread, NULL);
        s->send_thread_running = 0;
    }
    if (s->recv_thread_running) {
        atomic_store_explicit(&s->recv_thread_stop, 1, memory_order_release);
        pthread_join(s->recv_thread, NULL);
        s->recv_thread_running = 0;
    }
}

static void drain_queues(SpacemiTMPPDecContext *s)
{
    /* Drain send ring */
    int tail = atomic_load_explicit(&s->send_tail, memory_order_relaxed);
    int head = atomic_load_explicit(&s->send_head, memory_order_relaxed);
    while (tail != head) {
        av_free(s->send_data[tail]);
        tail = (tail + 1) % SEND_RING_SIZE;
    }
    atomic_store(&s->send_head, 0);
    atomic_store(&s->send_tail, 0);
    atomic_store(&s->send_eos, 0);

    /* Drain frame ring */
    tail = atomic_load_explicit(&s->frame_tail, memory_order_relaxed);
    head = atomic_load_explicit(&s->frame_head, memory_order_relaxed);
    while (tail != head) {
        DecodedFrame *df = &s->frame_ring[tail];
        av_free(df->y_data);
        av_free(df->uv_data);
        df->y_data = NULL;
        df->uv_data = NULL;
        tail = (tail + 1) % FRAME_RING_SIZE;
    }
    atomic_store(&s->frame_head, 0);
    atomic_store(&s->frame_tail, 0);
}

static void spacemit_decode_flush(AVCodecContext *avctx)
{
    SpacemiTMPPDecContext *s = avctx->priv_data;

    stop_threads(s);
    drain_queues(s);

    if (s->mpp_ctx) VDEC_Flush(s->mpp_ctx);
    s->draining = 0;
    atomic_store(&s->eos_received, 0);
    if (s->bsf) av_bsf_flush(s->bsf);

    /* Restart threads */
    atomic_store(&s->send_thread_stop, 0);
    atomic_store(&s->recv_thread_stop, 0);
    if (pthread_create(&s->recv_thread, NULL, recv_thread_func, s) == 0)
        s->recv_thread_running = 1;
    if (pthread_create(&s->send_thread, NULL, send_thread_func, s) == 0)
        s->send_thread_running = 1;
}

static int spacemit_decode_close(AVCodecContext *avctx)
{
    SpacemiTMPPDecContext *s = avctx->priv_data;

    stop_threads(s);
    drain_queues(s);

    if (s->mpp_frame) { FRAME_Destory(s->mpp_frame); s->mpp_frame = NULL; }
    if (s->mpp_pkt)   { PACKET_Destory(s->mpp_pkt);  s->mpp_pkt = NULL; }
    if (s->mpp_ctx)   { VDEC_DestoryChannel(s->mpp_ctx); s->mpp_ctx = NULL; }

    if (s->bsf) av_bsf_free(&s->bsf);
    return 0;
}

#define DEFINE_SPACEMIT_DECODER(SHORT, AVID)                                    \
static const AVClass spacemit_mpp_##SHORT##_dec_class = {                       \
    .class_name = #SHORT "_spacemit_mpp_decoder",                               \
    .item_name  = av_default_item_name,                                         \
    .version    = LIBAVUTIL_VERSION_INT,                                        \
};                                                                              \
const FFCodec ff_##SHORT##_spacemit_mpp_decoder = {                             \
    .p.name         = #SHORT "_spacemit_mpp",                                   \
    .p.long_name    = NULL_IF_CONFIG_SMALL("SpacemiT K1 MPP " #SHORT),         \
    .p.type         = AVMEDIA_TYPE_VIDEO,                                       \
    .p.id           = AVID,                                                     \
    .p.capabilities = AV_CODEC_CAP_DELAY,                                      \
    .priv_data_size = sizeof(SpacemiTMPPDecContext),                            \
    .p.priv_class   = &spacemit_mpp_##SHORT##_dec_class,                        \
    .init           = spacemit_decode_init,                                     \
    FF_CODEC_RECEIVE_FRAME_CB(spacemit_decode_receive_frame),                   \
    .flush          = spacemit_decode_flush,                                    \
    .close          = spacemit_decode_close,                                    \
    .caps_internal  = FF_CODEC_CAP_INIT_CLEANUP,                                \
}

DEFINE_SPACEMIT_DECODER(h264,  AV_CODEC_ID_H264);
DEFINE_SPACEMIT_DECODER(hevc,  AV_CODEC_ID_HEVC);
DEFINE_SPACEMIT_DECODER(vp8,   AV_CODEC_ID_VP8);
DEFINE_SPACEMIT_DECODER(mpeg2, AV_CODEC_ID_MPEG2VIDEO);
DEFINE_SPACEMIT_DECODER(mpeg4, AV_CODEC_ID_MPEG4);
