/*
 * spacemitmppenc.c — SpacemiT K1 MPP hardware encoder (FFmpeg codec plugin)
 *
 * Provides H.264, H.265, VP8, VP9 hardware encoding via the K1 SoC's
 * Media Processing Pipeline (MPP), exposed through the standard Linux
 * V4L2 M2M kernel interface.
 *
 * ── Pixel format handling ──────────────────────────────────────────────────
 *
 * This encoder accepts both yuv420p and nv12 frames.  The conversion from
 * yuv420p to the NV12 layout expected by the hardware is done inline with
 * plain C (the RVV-accelerated yuv420p_to_nv12 library is NOT required).
 * Benchmarking on the K1 shows libswscale at comparable speed, so the
 * external library adds no meaningful benefit for this path.
 *
 * ── K1 driver specifics (mvx / Linlon) ────────────────────────────────────
 *
 * The mvx driver is a MULTIPLANAR M2M device (V4L2_CAP_VIDEO_M2M_MPLANE).
 * All ioctls must use _MPLANE buffer types.  After VIDIOC_S_FMT we always
 * do VIDIOC_G_FMT to read back the actual num_planes the driver assigned:
 *   NV12 OUTPUT  → planes = 2  (Y plane + UV plane, separate mmap regions)
 *   H264 CAPTURE → planes = 1  (bitstream, single mmap region)
 *
 * ── Supported codecs ───────────────────────────────────────────────────────
 *   h264_spacemit_mpp, hevc_spacemit_mpp, vp8_spacemit_mpp, vp9_spacemit_mpp
 */

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#ifndef O_CLOEXEC
#  define O_CLOEXEC 02000000
#endif
#include <linux/videodev2.h>

#include "avcodec.h"
#include "codec_internal.h"
#include "encode.h"
#include "libavutil/log.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/avassert.h"
#include "libavutil/mem.h"
#include "libavutil/imgutils.h"
#include "libavutil/frame.h"

#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"

#include "spacemitv2d.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════════════════ */

#define SPACEMIT_MPP_OUTPUT_BUFS   4
#define SPACEMIT_MPP_CAPTURE_BUFS  4
#define SPACEMIT_MPP_CAPTURE_SIZE  (1024 * 1024 * 2)
#define SPACEMIT_MPP_MAX_PLANES    3

#define OUT_TYPE  V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE
#define CAP_TYPE  V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE

/* ═══════════════════════════════════════════════════════════════════════════
 * V4L2 mmap buffer descriptor
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct V4L2Buf {
    void   *start[SPACEMIT_MPP_MAX_PLANES];
    size_t  length[SPACEMIT_MPP_MAX_PLANES];
    int     num_planes;
    int     index;
    int     queued;
} V4L2Buf;

/* ═══════════════════════════════════════════════════════════════════════════
 * Private encoder context
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct SpacemitMppContext {
    const AVClass *class;

    /* ── AVOptions ── */
    char         *device;
    int           num_out_bufs;
    int           num_cap_bufs;

    /* ── V4L2 state ── */
    int           fd;
    __u32         v4l2_codec;

    int           out_num_planes;
    int           cap_num_planes;

    V4L2Buf      *out_bufs;
    int           n_out;

    V4L2Buf      *cap_bufs;
    int           n_cap;

    /* ── PTS reordering ── */
    int64_t       pts_queue[SPACEMIT_MPP_OUTPUT_BUFS * 2];
    int           pts_head;
    int           pts_tail;
    int64_t       last_emitted_pts;     /* monotonic PTS for synthesis fallback */

    /* ── V2D zero-copy: DMA-buf fds exported from OUTPUT buffers ── */
    int               out_dma_fds[SPACEMIT_MPP_OUTPUT_BUFS][SPACEMIT_MPP_MAX_PLANES];
    SpacemitV2DCtx    v2d;

    /* ── Stream state ── */
    int           started;
    int           draining;

    /* ── Encoder params ── */
    int           width;
    int           height;
    int           bitrate;
    int           gop_size;
    int           max_b_frames;
    char         *profile;
    char         *level;
} SpacemitMppContext;

/* ═══════════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

static int xioctl(int fd, unsigned long request, void *arg)
{
    int ret;
    do { ret = ioctl(fd, request, arg); } while (ret == -1 && errno == EINTR);
    return ret;
}

static __u32 avcodec_to_v4l2_codec(enum AVCodecID id)
{
    switch (id) {
    case AV_CODEC_ID_H264:  return V4L2_PIX_FMT_H264;
    case AV_CODEC_ID_HEVC:  return V4L2_PIX_FMT_HEVC;
    case AV_CODEC_ID_VP8:   return V4L2_PIX_FMT_VP8;
    case AV_CODEC_ID_VP9:   return V4L2_PIX_FMT_VP9;
    case AV_CODEC_ID_MJPEG: return V4L2_PIX_FMT_JPEG;
    default:                return 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Device probe
 * ═══════════════════════════════════════════════════════════════════════════ */

static int find_v4l2_device(AVCodecContext *avctx, __u32 v4l2_codec)
{
    SpacemitMppContext *s = avctx->priv_data;
    char path[32];
    int  i;

    for (i = 0; i < 16; i++) {
        snprintf(path, sizeof(path), "/dev/video%d", i);
        int fd = open(path, O_RDWR | O_CLOEXEC | O_NONBLOCK);
        if (fd < 0) continue;

        struct v4l2_capability cap = {0};
        if (xioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) { close(fd); continue; }

        if (!(cap.device_caps & V4L2_CAP_VIDEO_M2M_MPLANE) ||
            !(cap.device_caps & V4L2_CAP_STREAMING))       { close(fd); continue; }

        struct v4l2_fmtdesc fdesc = {0};
        fdesc.type = CAP_TYPE;
        int found  = 0;
        for (fdesc.index = 0; xioctl(fd, VIDIOC_ENUM_FMT, &fdesc) == 0; fdesc.index++) {
            if (fdesc.pixelformat == v4l2_codec) { found = 1; break; }
        }

        if (found) {
            av_log(avctx, AV_LOG_INFO,
                   "spacemit_mpp: using device %s (%s)\n", path, cap.card);
            s->fd = fd;
            return 0;
        }
        close(fd);
    }
    av_log(avctx, AV_LOG_ERROR, "spacemit_mpp: no suitable V4L2 M2M MPLANE encoder\n");
    return AVERROR(ENODEV);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Set formats — always use multiplanar types.
 * Read back num_planes with G_FMT after S_FMT; driver may differ from request.
 * ═══════════════════════════════════════════════════════════════════════════ */

static int set_formats_mplane(AVCodecContext *avctx)
{
    SpacemitMppContext *s = avctx->priv_data;

    /* OUTPUT: NV12 raw encoder input */
    {
        struct v4l2_format fmt = {0};
        fmt.type                              = OUT_TYPE;
        fmt.fmt.pix_mp.width                  = s->width;
        fmt.fmt.pix_mp.height                 = s->height;
        fmt.fmt.pix_mp.pixelformat            = V4L2_PIX_FMT_NV12;
        fmt.fmt.pix_mp.field                  = V4L2_FIELD_NONE;
        fmt.fmt.pix_mp.num_planes             = 1;
        fmt.fmt.pix_mp.plane_fmt[0].sizeimage    = s->width * s->height * 3 / 2;
        fmt.fmt.pix_mp.plane_fmt[0].bytesperline = s->width;

        if (xioctl(s->fd, VIDIOC_S_FMT, &fmt) < 0) {
            av_log(avctx, AV_LOG_ERROR,
                   "spacemit_mpp: S_FMT OUTPUT NV12 failed: %s\n", strerror(errno));
            return AVERROR(errno);
        }

        struct v4l2_format gfmt = {0};
        gfmt.type = OUT_TYPE;
        if (xioctl(s->fd, VIDIOC_G_FMT, &gfmt) < 0) {
            av_log(avctx, AV_LOG_ERROR,
                   "spacemit_mpp: G_FMT OUTPUT failed: %s\n", strerror(errno));
            return AVERROR(errno);
        }
        s->out_num_planes = gfmt.fmt.pix_mp.num_planes;
        av_log(avctx, AV_LOG_INFO,
               "spacemit_mpp: OUTPUT %dx%d pix=0x%x planes=%d\n",
               gfmt.fmt.pix_mp.width, gfmt.fmt.pix_mp.height,
               gfmt.fmt.pix_mp.pixelformat, s->out_num_planes);
    }

    /* CAPTURE: compressed bitstream */
    {
        struct v4l2_format fmt = {0};
        fmt.type                              = CAP_TYPE;
        fmt.fmt.pix_mp.width                  = s->width;
        fmt.fmt.pix_mp.height                 = s->height;
        fmt.fmt.pix_mp.pixelformat            = s->v4l2_codec;
        fmt.fmt.pix_mp.field                  = V4L2_FIELD_NONE;
        fmt.fmt.pix_mp.num_planes             = 1;
        fmt.fmt.pix_mp.plane_fmt[0].sizeimage = SPACEMIT_MPP_CAPTURE_SIZE;

        if (xioctl(s->fd, VIDIOC_S_FMT, &fmt) < 0) {
            av_log(avctx, AV_LOG_ERROR,
                   "spacemit_mpp: S_FMT CAPTURE failed: %s\n", strerror(errno));
            return AVERROR(errno);
        }

        struct v4l2_format gfmt = {0};
        gfmt.type = CAP_TYPE;
        if (xioctl(s->fd, VIDIOC_G_FMT, &gfmt) < 0) {
            av_log(avctx, AV_LOG_ERROR,
                   "spacemit_mpp: G_FMT CAPTURE failed: %s\n", strerror(errno));
            return AVERROR(errno);
        }
        s->cap_num_planes = gfmt.fmt.pix_mp.num_planes;
        av_log(avctx, AV_LOG_INFO,
               "spacemit_mpp: CAPTURE %dx%d pix=0x%x planes=%d\n",
               gfmt.fmt.pix_mp.width, gfmt.fmt.pix_mp.height,
               gfmt.fmt.pix_mp.pixelformat, s->cap_num_planes);
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Encoder controls
 * ═══════════════════════════════════════════════════════════════════════════ */

static void set_encoder_controls(AVCodecContext *avctx)
{
    SpacemitMppContext *s = avctx->priv_data;
    struct v4l2_ext_controls ctrls;
    struct v4l2_ext_control  ctrl;

#define SET_CTRL(ctrl_id, val) do {                              \
    memset(&ctrls, 0, sizeof(ctrls));                            \
    memset(&ctrl,  0, sizeof(ctrl));                             \
    ctrl.id        = (__u32)(ctrl_id);                           \
    ctrl.value     = (int32_t)(val);                             \
    ctrls.which    = V4L2_CTRL_WHICH_CUR_VAL;                    \
    ctrls.count    = 1;                                          \
    ctrls.controls = &ctrl;                                      \
    if (xioctl(s->fd, VIDIOC_S_EXT_CTRLS, &ctrls) < 0)          \
        av_log(avctx, AV_LOG_WARNING,                            \
               "spacemit_mpp: failed to set ctrl 0x%x\n",        \
               (ctrl_id));                                       \
} while (0)

    if (s->bitrate > 0)
        SET_CTRL(V4L2_CID_MPEG_VIDEO_BITRATE, s->bitrate);

    if (avctx->codec_id == AV_CODEC_ID_H264) {
        if (s->gop_size > 0)
            SET_CTRL(V4L2_CID_MPEG_VIDEO_H264_I_PERIOD, s->gop_size);
        if (s->max_b_frames == 0)
            SET_CTRL(V4L2_CID_MPEG_VIDEO_B_FRAMES, 0);
        SET_CTRL(V4L2_CID_MPEG_VIDEO_H264_PROFILE,
                 V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE);
    }

    if (avctx->codec_id == AV_CODEC_ID_HEVC) {
        if (s->gop_size > 0)
            SET_CTRL(V4L2_CID_MPEG_VIDEO_HEVC_I_FRAME_QP, s->gop_size);
    }

#undef SET_CTRL
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Allocate & mmap multiplanar buffers
 * ═══════════════════════════════════════════════════════════════════════════ */

static int alloc_buffers(AVCodecContext *avctx,
                         __u32 buf_type, int num_planes,
                         V4L2Buf **bufs_out, int *n_out, int count)
{
    SpacemitMppContext *s = avctx->priv_data;
    struct v4l2_requestbuffers req = {0};
    req.type   = buf_type;
    req.memory = V4L2_MEMORY_MMAP;
    req.count  = count;

    if (xioctl(s->fd, VIDIOC_REQBUFS, &req) < 0) {
        av_log(avctx, AV_LOG_ERROR,
               "spacemit_mpp: REQBUFS type=%u failed: %s\n",
               buf_type, strerror(errno));
        return AVERROR(errno);
    }

    *n_out    = req.count;
    *bufs_out = av_calloc(req.count, sizeof(V4L2Buf));
    if (!*bufs_out) return AVERROR(ENOMEM);

    for (int i = 0; i < (int)req.count; i++) {
        struct v4l2_plane  planes[SPACEMIT_MPP_MAX_PLANES] = {};
        struct v4l2_buffer vbuf = {0};
        vbuf.type     = buf_type;
        vbuf.memory   = V4L2_MEMORY_MMAP;
        vbuf.index    = i;
        vbuf.m.planes = planes;
        vbuf.length   = num_planes;

        if (xioctl(s->fd, VIDIOC_QUERYBUF, &vbuf) < 0) {
            av_log(avctx, AV_LOG_ERROR,
                   "spacemit_mpp: QUERYBUF i=%d np=%d failed: %s\n",
                   i, num_planes, strerror(errno));
            return AVERROR(errno);
        }

        (*bufs_out)[i].index      = i;
        (*bufs_out)[i].num_planes = num_planes;

        for (int p = 0; p < num_planes; p++) {
            (*bufs_out)[i].length[p] = planes[p].length;
            (*bufs_out)[i].start[p]  = mmap(NULL, planes[p].length,
                                            PROT_READ | PROT_WRITE, MAP_SHARED,
                                            s->fd, planes[p].m.mem_offset);
            if ((*bufs_out)[i].start[p] == MAP_FAILED) {
                av_log(avctx, AV_LOG_ERROR,
                       "spacemit_mpp: mmap i=%d p=%d failed: %s\n",
                       i, p, strerror(errno));
                return AVERROR(errno);
            }
        }
    }
    return 0;
}

static void free_buffers(V4L2Buf *bufs, int n)
{
    if (!bufs) return;
    for (int i = 0; i < n; i++)
        for (int p = 0; p < bufs[i].num_planes; p++)
            if (bufs[i].start[p] && bufs[i].start[p] != MAP_FAILED)
                munmap(bufs[i].start[p], bufs[i].length[p]);
    av_free(bufs);
}

static int enqueue_all_capture(AVCodecContext *avctx)
{
    SpacemitMppContext *s = avctx->priv_data;

    for (int i = 0; i < s->n_cap; i++) {
        struct v4l2_plane  planes[SPACEMIT_MPP_MAX_PLANES] = {};
        struct v4l2_buffer vbuf = {0};
        vbuf.type     = CAP_TYPE;
        vbuf.memory   = V4L2_MEMORY_MMAP;
        vbuf.index    = i;
        vbuf.m.planes = planes;
        vbuf.length   = s->cap_num_planes;
        for (int p = 0; p < s->cap_num_planes; p++) {
            planes[p].length    = s->cap_bufs[i].length[p];
            planes[p].bytesused = 0;
        }
        if (xioctl(s->fd, VIDIOC_QBUF, &vbuf) < 0) {
            av_log(avctx, AV_LOG_ERROR,
                   "spacemit_mpp: pre-QBUF CAPTURE i=%d failed: %s\n",
                   i, strerror(errno));
            return AVERROR(errno);
        }
        s->cap_bufs[i].queued = 1;
    }
    return 0;
}

static int stream_on(AVCodecContext *avctx)
{
    SpacemitMppContext *s = avctx->priv_data;
    __u32 type = OUT_TYPE;
    if (xioctl(s->fd, VIDIOC_STREAMON, &type) < 0) {
        av_log(avctx, AV_LOG_ERROR,
               "spacemit_mpp: STREAMON OUTPUT failed: %s\n", strerror(errno));
        return AVERROR(errno);
    }
    type = CAP_TYPE;
    if (xioctl(s->fd, VIDIOC_STREAMON, &type) < 0) {
        av_log(avctx, AV_LOG_ERROR,
               "spacemit_mpp: STREAMON CAPTURE failed: %s\n", strerror(errno));
        return AVERROR(errno);
    }
    return 0;
}

static void stream_off(SpacemitMppContext *s)
{
    if (s->fd < 0) return;
    __u32 type = OUT_TYPE;
    xioctl(s->fd, VIDIOC_STREAMOFF, &type);
    type = CAP_TYPE;
    xioctl(s->fd, VIDIOC_STREAMOFF, &type);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PTS queue
 * ═══════════════════════════════════════════════════════════════════════════ */

#define PTS_QUEUE_MASK  (SPACEMIT_MPP_OUTPUT_BUFS * 2 - 1)

static void pts_push(SpacemitMppContext *s, int64_t pts)
{
    s->pts_queue[s->pts_head & PTS_QUEUE_MASK] = pts;
    s->pts_head++;
}

static int64_t pts_pop(SpacemitMppContext *s)
{
    if (s->pts_head == s->pts_tail) return AV_NOPTS_VALUE;
    int64_t pts = s->pts_queue[s->pts_tail & PTS_QUEUE_MASK];
    s->pts_tail++;
    return pts;
}

static int get_free_output_buf(SpacemitMppContext *s)
{
    for (int i = 0; i < s->n_out; i++)
        if (!s->out_bufs[i].queued) return i;
    return -1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Copy frame data into V4L2 OUTPUT buffer planes
 *
 * The mvx driver reports 2 planes for NV12 (Y + UV as separate mmap regions).
 * For yuv420p input we interleave U and V inline — no external library needed.
 *
 * num_planes = 1: contiguous NV12  [Y][UV]  (single mmap region)
 * num_planes = 2: separate NM12    plane0=Y, plane1=UV
 * num_planes = 3: I420 fallback    plane0=Y, plane1=U, plane2=V
 * ═══════════════════════════════════════════════════════════════════════════ */

static void copy_yuv420p_to_planes(V4L2Buf *buf,
                                   const uint8_t *y_src, int y_stride,
                                   const uint8_t *u_src, int u_stride,
                                   const uint8_t *v_src,
                                   int width, int height)
{
    int np   = buf->num_planes;
    int uv_h = height / 2;
    int uv_w = width  / 2;
    int row, col;

    /* ── Y plane (always plane 0) ── */
    {
        uint8_t *dst = (uint8_t *)buf->start[0];
        if (y_stride == width) {
            memcpy(dst, y_src, (size_t)width * height);
        } else {
            for (row = 0; row < height; row++)
                memcpy(dst + row * width, y_src + row * y_stride, width);
        }
    }

    if (np == 1) {
        /* Contiguous: interleaved UV immediately after Y */
        uint8_t *dst_uv = (uint8_t *)buf->start[0] + width * height;
        for (row = 0; row < uv_h; row++) {
            const uint8_t *us = u_src + row * u_stride;
            const uint8_t *vs = v_src + row * u_stride;
            uint8_t       *d  = dst_uv + row * width;
            for (col = 0; col < uv_w; col++) {
                d[col * 2]     = us[col];
                d[col * 2 + 1] = vs[col];
            }
        }
    } else if (np == 2) {
        /* Separate UV plane */
        uint8_t *dst_uv = (uint8_t *)buf->start[1];
        for (row = 0; row < uv_h; row++) {
            const uint8_t *us = u_src + row * u_stride;
            const uint8_t *vs = v_src + row * u_stride;
            uint8_t       *d  = dst_uv + row * width;
            for (col = 0; col < uv_w; col++) {
                d[col * 2]     = us[col];
                d[col * 2 + 1] = vs[col];
            }
        }
    } else {
        /* I420: separate U and V planes */
        if (np >= 2) {
            uint8_t *dst_u = (uint8_t *)buf->start[1];
            if (u_stride == uv_w)
                memcpy(dst_u, u_src, (size_t)uv_w * uv_h);
            else
                for (row = 0; row < uv_h; row++)
                    memcpy(dst_u + row * uv_w, u_src + row * u_stride, uv_w);
        }
        if (np >= 3) {
            uint8_t *dst_v = (uint8_t *)buf->start[2];
            if (u_stride == uv_w)
                memcpy(dst_v, v_src, (size_t)uv_w * uv_h);
            else
                for (row = 0; row < uv_h; row++)
                    memcpy(dst_v + row * uv_w, v_src + row * u_stride, uv_w);
        }
    }
}

static void copy_nv12_to_planes(V4L2Buf *buf,
                                const uint8_t *y_src,  int y_stride,
                                const uint8_t *uv_src, int uv_stride,
                                int width, int height)
{
    int np   = buf->num_planes;
    int uv_h = height / 2;
    int row;

    /* Y */
    {
        uint8_t *dst = (uint8_t *)buf->start[0];
        if (y_stride == width)
            memcpy(dst, y_src, (size_t)width * height);
        else
            for (row = 0; row < height; row++)
                memcpy(dst + row * width, y_src + row * y_stride, width);
    }

    /* UV */
    {
        uint8_t *dst_uv = (np == 1)
            ? (uint8_t *)buf->start[0] + width * height
            : (uint8_t *)buf->start[1];
        if (uv_stride == width)
            memcpy(dst_uv, uv_src, (size_t)width * uv_h);
        else
            for (row = 0; row < uv_h; row++)
                memcpy(dst_uv + row * width, uv_src + row * uv_stride, width);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * av_cold init
 * ═══════════════════════════════════════════════════════════════════════════ */

static av_cold int spacemit_mpp_encode_init(AVCodecContext *avctx)
{
    SpacemitMppContext *s = avctx->priv_data;
    s->last_emitted_pts = -1;
    int ret;

    s->fd = -1;

    s->v4l2_codec = avcodec_to_v4l2_codec(avctx->codec_id);
    if (!s->v4l2_codec) {
        av_log(avctx, AV_LOG_ERROR,
               "spacemit_mpp: unsupported codec id %d\n", avctx->codec_id);
        return AVERROR(EINVAL);
    }

    if (s->device && s->device[0]) {
        s->fd = open(s->device, O_RDWR | O_CLOEXEC | O_NONBLOCK);
        if (s->fd < 0) {
            av_log(avctx, AV_LOG_ERROR,
                   "spacemit_mpp: open(%s) failed: %s\n",
                   s->device, strerror(errno));
            return AVERROR(errno);
        }
    } else {
        ret = find_v4l2_device(avctx, s->v4l2_codec);
        if (ret < 0) return ret;
    }

    s->width        = avctx->width;
    s->height       = avctx->height;
    s->bitrate      = (int)avctx->bit_rate;
    s->gop_size     = avctx->gop_size    > 0 ? avctx->gop_size    : 30;
    s->max_b_frames = avctx->max_b_frames;

    ret = set_formats_mplane(avctx);
    if (ret < 0) goto fail;

    set_encoder_controls(avctx);

    ret = alloc_buffers(avctx, OUT_TYPE, s->out_num_planes,
                        &s->out_bufs, &s->n_out,
                        s->num_out_bufs > 0 ? s->num_out_bufs
                                            : SPACEMIT_MPP_OUTPUT_BUFS);
    if (ret < 0) goto fail;

    /* Export encoder OUTPUT buffer planes as DMA-buf fds for V2D */
    memset(s->out_dma_fds, -1, sizeof(s->out_dma_fds));
    for (int _i = 0; _i < s->n_out && _i < SPACEMIT_MPP_OUTPUT_BUFS; _i++) {
        for (int _p = 0; _p < s->out_num_planes; _p++) {
            struct v4l2_exportbuffer expbuf;
            memset(&expbuf, 0, sizeof(expbuf));
            expbuf.type  = OUT_TYPE;
            expbuf.index = (unsigned)_i;
            expbuf.plane = (unsigned)_p;
            expbuf.flags = O_CLOEXEC;
            if (xioctl(s->fd, VIDIOC_EXPBUF, &expbuf) == 0)
                s->out_dma_fds[_i][_p] = expbuf.fd;
        }
    }
    spacemit_v2d_init(&s->v2d, avctx);   /* non-fatal if V2D unavailable */

    ret = alloc_buffers(avctx, CAP_TYPE, s->cap_num_planes,
                        &s->cap_bufs, &s->n_cap,
                        s->num_cap_bufs > 0 ? s->num_cap_bufs
                                            : SPACEMIT_MPP_CAPTURE_BUFS);
    if (ret < 0) goto fail;

    ret = enqueue_all_capture(avctx);
    if (ret < 0) goto fail;

    ret = stream_on(avctx);
    if (ret < 0) goto fail;

    s->started = 1;

    av_log(avctx, AV_LOG_INFO,
           "spacemit_mpp: encoder ready — %dx%d bitrate=%d gop=%d "
           "out_planes=%d cap_planes=%d\n",
           s->width, s->height, s->bitrate, s->gop_size,
           s->out_num_planes, s->cap_num_planes);

    avctx->pix_fmt = AV_PIX_FMT_YUV420P;
    return 0;

fail:
    stream_off(s);
    free_buffers(s->out_bufs, s->n_out); s->out_bufs = NULL;
    free_buffers(s->cap_bufs, s->n_cap); s->cap_bufs = NULL;
    /* Close V2D DMA-buf fds */
    for (int _i = 0; _i < SPACEMIT_MPP_OUTPUT_BUFS; _i++)
        for (int _p = 0; _p < SPACEMIT_MPP_MAX_PLANES; _p++)
            if (s->out_dma_fds[_i][_p] >= 0)
                close(s->out_dma_fds[_i][_p]);
    spacemit_v2d_close(&s->v2d);
    if (s->fd >= 0) { close(s->fd); s->fd = -1; }
    return ret;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * send_frame
 * ═══════════════════════════════════════════════════════════════════════════ */

static int spacemit_mpp_send_frame(AVCodecContext *avctx, const AVFrame *frame)
{
    SpacemitMppContext *s = avctx->priv_data;

    if (!frame) {
        if (!s->draining) {
            s->draining = 1;
            av_log(avctx, AV_LOG_DEBUG, "spacemit_mpp: EOS\n");
        }
        return 0;
    }

    int bi = get_free_output_buf(s);
    if (bi < 0) return AVERROR(EAGAIN);

    /* ── V2D zero-copy path ─────────────────────────────────────────────
     * If the frame came from the SpacemiT hardware decoder it carries
     * DMA-buf fds in frame->opaque.  Use V2D to blit directly from the
     * decoder's CAPTURE buffer into our OUTPUT buffer — no CPU involved.
     * Falls back to CPU memcpy if V2D is unavailable or opaque is unset.
     * ─────────────────────────────────────────────────────────────────── */
    int v2d_used = 0;
    if (frame->opaque && s->v2d.available &&
        frame->format == AV_PIX_FMT_NV12) {
        const SpacemitDMAFrameInfo *dma =
            (const SpacemitDMAFrameInfo *)frame->opaque;
        if (dma->dma_fds[0] >= 0) {
            int ret_v2d = spacemit_v2d_blit(
                &s->v2d, dma,
                s->out_dma_fds[bi], s->out_num_planes,
                s->width, s->height, s->width);
            if (ret_v2d == 0) {
                v2d_used = 1;
                av_log(avctx, AV_LOG_DEBUG,
                       "spacemit_mpp: V2D blit buf=%d\n", bi);
            } else {
                av_log(avctx, AV_LOG_WARNING,
                       "spacemit_mpp: V2D blit failed, using CPU fallback\n");
            }
        }
    }

    if (!v2d_used) {
        /* ── CPU fallback path ── */
        if (frame->format == AV_PIX_FMT_YUV420P) {
            copy_yuv420p_to_planes(&s->out_bufs[bi],
                frame->data[0], frame->linesize[0],
                frame->data[1], frame->linesize[1],
                frame->data[2],
                s->width, s->height);
        } else if (frame->format == AV_PIX_FMT_NV12) {
            copy_nv12_to_planes(&s->out_bufs[bi],
                frame->data[0], frame->linesize[0],
                frame->data[1], frame->linesize[1],
                s->width, s->height);
        } else {
            av_log(avctx, AV_LOG_ERROR,
                   "spacemit_mpp: unsupported pixel format %s\n",
                   av_get_pix_fmt_name(frame->format));
            return AVERROR(EINVAL);
        }
    }

    struct v4l2_plane  planes[SPACEMIT_MPP_MAX_PLANES] = {};
    struct v4l2_buffer vbuf = {0};
    vbuf.type     = OUT_TYPE;
    vbuf.memory   = V4L2_MEMORY_MMAP;
    vbuf.index    = bi;
    vbuf.m.planes = planes;
    vbuf.length   = s->out_num_planes;
    vbuf.timestamp.tv_sec  = (long)(frame->pts / 1000000);
    vbuf.timestamp.tv_usec = (long)(frame->pts % 1000000);

    if (s->out_num_planes == 1) {
        planes[0].bytesused = (__u32)(s->width * s->height * 3 / 2);
        planes[0].length    = (__u32)s->out_bufs[bi].length[0];
    } else if (s->out_num_planes >= 2) {
        planes[0].bytesused = (__u32)(s->width * s->height);
        planes[0].length    = (__u32)s->out_bufs[bi].length[0];
        planes[1].bytesused = (__u32)(s->width * s->height / 2);
        planes[1].length    = (__u32)s->out_bufs[bi].length[1];
    }

    if (xioctl(s->fd, VIDIOC_QBUF, &vbuf) < 0) {
        av_log(avctx, AV_LOG_ERROR,
               "spacemit_mpp: QBUF OUTPUT i=%d failed: %s\n",
               bi, strerror(errno));
        return AVERROR(errno);
    }
    /* Release DMA frame info allocated by decoder (if any) */
    if (frame->opaque) {
        av_freep(&frame->opaque);
    }

    s->out_bufs[bi].queued = 1;
    pts_push(s, frame->pts);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * receive_packet
 * ═══════════════════════════════════════════════════════════════════════════ */

static int spacemit_mpp_receive_packet(AVCodecContext *avctx, AVPacket *pkt)
{
    SpacemitMppContext *s = avctx->priv_data;

    /* Return completed OUTPUT buf to free pool */
    {
        struct v4l2_plane  planes[SPACEMIT_MPP_MAX_PLANES] = {};
        struct v4l2_buffer vbuf = {0};
        vbuf.type     = OUT_TYPE;
        vbuf.memory   = V4L2_MEMORY_MMAP;
        vbuf.m.planes = planes;
        vbuf.length   = s->out_num_planes;
        if (xioctl(s->fd, VIDIOC_DQBUF, &vbuf) == 0)
            s->out_bufs[vbuf.index].queued = 0;
    }

    /* Dequeue encoded packet */
    struct v4l2_plane  cap_planes[SPACEMIT_MPP_MAX_PLANES] = {};
    struct v4l2_buffer cap_buf = {0};
    cap_buf.type     = CAP_TYPE;
    cap_buf.memory   = V4L2_MEMORY_MMAP;
    cap_buf.m.planes = cap_planes;
    cap_buf.length   = s->cap_num_planes;

    if (xioctl(s->fd, VIDIOC_DQBUF, &cap_buf) < 0) {
        if (errno == EAGAIN)
            return s->draining ? AVERROR_EOF : AVERROR(EAGAIN);
        av_log(avctx, AV_LOG_ERROR,
               "spacemit_mpp: DQBUF CAPTURE failed: %s\n", strerror(errno));
        return AVERROR(errno);
    }

    __u32 pkt_size = cap_planes[0].bytesused;
    int   ret      = av_new_packet(pkt, (int)pkt_size);
    if (ret < 0) goto requeue;

    memcpy(pkt->data, s->cap_bufs[cap_buf.index].start[0], pkt_size);

    /* PTS recovery, in priority order:
     * 1. V4L2 driver propagated the input timestamp to the output buffer.
     * 2. The pts FIFO populated on QBUF.
     * 3. Synthesised: last_emitted_pts + 1 - guarantees monotonicity for
     *    HLS/fmp4 muxers. Covers the V4L2 stateful encoder's initial
     *    header packet (VPS/SPS/PPS) which has no input frame association. */
    int64_t v4l2_pts = (int64_t)cap_buf.timestamp.tv_sec * 1000000
                     + (int64_t)cap_buf.timestamp.tv_usec;
    int64_t queued = pts_pop(s);
    if (cap_buf.timestamp.tv_sec != 0 || cap_buf.timestamp.tv_usec != 0) {
        pkt->pts = v4l2_pts;
    } else if (queued != AV_NOPTS_VALUE) {
        pkt->pts = queued;
    } else {
        pkt->pts = s->last_emitted_pts + 1;
    }
    if (pkt->pts <= s->last_emitted_pts)
        pkt->pts = s->last_emitted_pts + 1;
    s->last_emitted_pts = pkt->pts;
    pkt->dts = pkt->pts;

    if (cap_buf.flags & V4L2_BUF_FLAG_KEYFRAME)
        pkt->flags |= AV_PKT_FLAG_KEY;

requeue:
    {
        struct v4l2_plane rq[SPACEMIT_MPP_MAX_PLANES] = {};
        for (int p = 0; p < s->cap_num_planes; p++) {
            rq[p].length    = s->cap_bufs[cap_buf.index].length[p];
            rq[p].bytesused = 0;
        }
        cap_buf.m.planes = rq;
        cap_buf.length   = s->cap_num_planes;
        cap_buf.flags    = 0;
        if (xioctl(s->fd, VIDIOC_QBUF, &cap_buf) < 0)
            av_log(avctx, AV_LOG_WARNING,
                   "spacemit_mpp: re-QBUF CAPTURE %d failed: %s\n",
                   cap_buf.index, strerror(errno));
    }
    return ret;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * av_cold close
 * ═══════════════════════════════════════════════════════════════════════════ */

static av_cold int spacemit_mpp_encode_close(AVCodecContext *avctx)
{
    SpacemitMppContext *s = avctx->priv_data;
    stream_off(s);
    free_buffers(s->out_bufs, s->n_out); s->out_bufs = NULL;
    free_buffers(s->cap_bufs, s->n_cap); s->cap_bufs = NULL;
    /* Close V2D DMA-buf fds */
    for (int _i = 0; _i < SPACEMIT_MPP_OUTPUT_BUFS; _i++)
        for (int _p = 0; _p < SPACEMIT_MPP_MAX_PLANES; _p++)
            if (s->out_dma_fds[_i][_p] >= 0)
                close(s->out_dma_fds[_i][_p]);
    spacemit_v2d_close(&s->v2d);
    if (s->fd >= 0) { close(s->fd); s->fd = -1; }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Pixel formats, options, FFCodec registration
 * ═══════════════════════════════════════════════════════════════════════════ */

static const enum AVPixelFormat spacemit_mpp_pix_fmts[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_NONE
};

static const AVOption spacemit_mpp_options[] = {
    { "device",
      "V4L2 device node (empty = auto-detect)",
      offsetof(SpacemitMppContext, device),
      AV_OPT_TYPE_STRING, { .str = "" }, 0, 0,
      AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM },
    { "num_output_bufs",
      "number of V4L2 OUTPUT queue buffers",
      offsetof(SpacemitMppContext, num_out_bufs),
      AV_OPT_TYPE_INT, { .i64 = SPACEMIT_MPP_OUTPUT_BUFS }, 2, 16,
      AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM },
    { "num_capture_bufs",
      "number of V4L2 CAPTURE queue buffers",
      offsetof(SpacemitMppContext, num_cap_bufs),
      AV_OPT_TYPE_INT, { .i64 = SPACEMIT_MPP_CAPTURE_BUFS }, 2, 16,
      AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM },
    { "profile",
      "Set the encoding profile (accepted but ignored)",
      offsetof(SpacemitMppContext, profile),
      AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0,
      AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM },
    { "level",
      "Set the encoding level (accepted but ignored)",
      offsetof(SpacemitMppContext, level),
      AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0,
      AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM },
    { NULL }
};

static int spacemit_mpp_receive_packet_wrapped(AVCodecContext *avctx, AVPacket *pkt)
{
    AVFrame *frame = av_frame_alloc();
    int got_frame, ret;
    if (!frame) return AVERROR(ENOMEM);

    got_frame = ff_encode_get_frame(avctx, frame);
    if (got_frame == 0) {
        ret = spacemit_mpp_send_frame(avctx, frame);
        av_frame_free(&frame);
        if (ret < 0 && ret != AVERROR(EAGAIN)) return ret;
    } else if (got_frame == AVERROR_EOF) {
        av_frame_free(&frame);
        spacemit_mpp_send_frame(avctx, NULL);
    } else {
        av_frame_free(&frame);
    }
    return spacemit_mpp_receive_packet(avctx, pkt);
}

#define DEFINE_SPACEMIT_MPP_ENCODER(SHORT, FULLNAME, AVID)                   \
static const AVClass spacemit_mpp_##SHORT##_class = {                        \
    .class_name = #SHORT "_spacemit_mpp",                                    \
    .item_name  = av_default_item_name,                                      \
    .option     = spacemit_mpp_options,                                      \
    .version    = LIBAVUTIL_VERSION_INT,                                     \
};                                                                            \
const FFCodec ff_##SHORT##_spacemit_mpp_encoder = {                          \
    .p.name         = #SHORT "_spacemit_mpp",                                \
    CODEC_LONG_NAME(FULLNAME " (SpacemiT K1 MPP, V4L2 M2M)"),               \
    .p.type         = AVMEDIA_TYPE_VIDEO,                                    \
    .p.id           = AVID,                                                  \
    .priv_data_size = sizeof(SpacemitMppContext),                            \
    .p.priv_class   = &spacemit_mpp_##SHORT##_class,                         \
    .init           = spacemit_mpp_encode_init,                              \
    FF_CODEC_RECEIVE_PACKET_CB(spacemit_mpp_receive_packet_wrapped),         \
    .close          = spacemit_mpp_encode_close,                             \
    .p.pix_fmts     = spacemit_mpp_pix_fmts,                                \
    .p.capabilities = AV_CODEC_CAP_HARDWARE | AV_CODEC_CAP_DR1,             \
    .p.wrapper_name = "spacemit_mpp",                                        \
    .caps_internal  = FF_CODEC_CAP_NOT_INIT_THREADSAFE,                     \
};

DEFINE_SPACEMIT_MPP_ENCODER(h264,  "H.264 / AVC",  AV_CODEC_ID_H264)
DEFINE_SPACEMIT_MPP_ENCODER(hevc,  "H.265 / HEVC", AV_CODEC_ID_HEVC)
DEFINE_SPACEMIT_MPP_ENCODER(vp8,   "VP8",          AV_CODEC_ID_VP8)
DEFINE_SPACEMIT_MPP_ENCODER(vp9,   "VP9",          AV_CODEC_ID_VP9)
DEFINE_SPACEMIT_MPP_ENCODER(mjpeg, "MJPEG",        AV_CODEC_ID_MJPEG)
