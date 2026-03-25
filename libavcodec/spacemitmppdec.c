/*
 * spacemitmppdec.c — SpacemiT K1 MPP hardware decoder (FFmpeg codec plugin)
 *
 * Provides H.264, H.265, VP8, VP9 hardware decoding via the K1 SoC's
 * Media Processing Pipeline (MPP), exposed through the standard Linux
 * V4L2 M2M kernel interface.
 *
 * ── K1 driver specifics (mvx / Linlon) ────────────────────────────────────
 *
 * The mvx driver is a MULTIPLANAR M2M device (V4L2_CAP_VIDEO_M2M_MPLANE).
 * All ioctls must use _MPLANE buffer types.
 *
 * Decode direction:
 *   OUTPUT  (V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE)  — compressed bitstream input
 *   CAPTURE (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) — decoded NV12 frames output
 *
 * After VIDIOC_S_FMT on the CAPTURE side the driver may change num_planes,
 * so we always do VIDIOC_G_FMT to read back the actual plane count:
 *   H264 OUTPUT  → planes = 1  (bitstream, single mmap region)
 *   NV12 CAPTURE → planes = 2  (Y plane + UV plane, separate mmap regions)
 *
 * ── Supported codecs ───────────────────────────────────────────────────────
 *   h264_spacemit_mpp, hevc_spacemit_mpp, vp8_spacemit_mpp, vp9_spacemit_mpp
 *
 * ── Dynamic resolution change ─────────────────────────────────────────────
 *
 * When the driver signals V4L2_EVENT_SOURCE_CHANGE (or returns EPIPE on
 * CAPTURE dequeue) we stop/restart the CAPTURE queue with updated geometry.
 * This is the standard V4L2 stateful codec resolution-change sequence.
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
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
#include "decode.h"
#include "libavutil/log.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/avassert.h"
#include "libavutil/mem.h"
#include "libavutil/imgutils.h"
#include "libavutil/frame.h"

#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"

/* ═══════════════════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════════════════ */

#define SPACEMIT_DEC_OUTPUT_BUFS   8   /* compressed input ring */
#define SPACEMIT_DEC_CAPTURE_BUFS  8   /* decoded frame pool */
#define SPACEMIT_DEC_MAX_PLANES    3

#define OUT_TYPE  V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE
#define CAP_TYPE  V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE

/* ═══════════════════════════════════════════════════════════════════════════
 * V4L2 mmap buffer descriptor
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct DecV4L2Buf {
    void   *start[SPACEMIT_DEC_MAX_PLANES];
    size_t  length[SPACEMIT_DEC_MAX_PLANES];
    int     num_planes;
    int     index;
    int     queued;
} DecV4L2Buf;

/* ═══════════════════════════════════════════════════════════════════════════
 * Private decoder context
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct SpacemitMppDecContext {
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

    DecV4L2Buf   *out_bufs;
    int           n_out;

    DecV4L2Buf   *cap_bufs;
    int           n_cap;

    /* ── Output geometry (may change after source_change event) ── */
    int           width;
    int           height;
    int           coded_width;
    int           coded_height;

    /* ── Stream state ── */
    int           output_started;
    int           capture_started;
    int           draining;
    int           eof;

    /* ── PTS queue (maps OUTPUT buf index → pts) ── */
    int64_t       pts_queue[SPACEMIT_DEC_OUTPUT_BUFS * 2];

} SpacemitMppDecContext;

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
    default:                return 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Device probe — find the first V4L2 M2M MPLANE device supporting our codec
 * ═══════════════════════════════════════════════════════════════════════════ */

static int find_v4l2_device(AVCodecContext *avctx, __u32 v4l2_codec)
{
    SpacemitMppDecContext *s = avctx->priv_data;
    char path[32];
    int  i;

    if (s->device && s->device[0]) {
        s->fd = open(s->device, O_RDWR | O_CLOEXEC | O_NONBLOCK);
        if (s->fd < 0) {
            av_log(avctx, AV_LOG_ERROR,
                   "spacemit_mpp: cannot open %s: %s\n",
                   s->device, strerror(errno));
            return AVERROR(errno);
        }
        return 0;
    }

    for (i = 0; i < 16; i++) {
        snprintf(path, sizeof(path), "/dev/video%d", i);
        int fd = open(path, O_RDWR | O_CLOEXEC | O_NONBLOCK);
        if (fd < 0) continue;

        struct v4l2_capability cap = {0};
        if (xioctl(fd, VIDIOC_QUERYCAP, &cap) < 0) { close(fd); continue; }

        if (!(cap.device_caps & V4L2_CAP_VIDEO_M2M_MPLANE) ||
            !(cap.device_caps & V4L2_CAP_STREAMING))       { close(fd); continue; }

        /* Check that the driver supports our codec on the OUTPUT side */
        struct v4l2_fmtdesc fdesc = {0};
        fdesc.type = OUT_TYPE;
        int found  = 0;
        for (fdesc.index = 0; xioctl(fd, VIDIOC_ENUM_FMT, &fdesc) == 0; fdesc.index++) {
            if (fdesc.pixelformat == v4l2_codec) { found = 1; break; }
        }

        if (found) {
            av_log(avctx, AV_LOG_INFO,
                   "spacemit_mpp dec: using device %s (%s)\n", path, cap.card);
            s->fd = fd;
            return 0;
        }
        close(fd);
    }
    av_log(avctx, AV_LOG_ERROR, "spacemit_mpp: no suitable V4L2 M2M MPLANE decoder\n");
    return AVERROR(ENODEV);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Set formats
 * ═══════════════════════════════════════════════════════════════════════════ */

static int set_formats_mplane(AVCodecContext *avctx)
{
    SpacemitMppDecContext *s = avctx->priv_data;
    int w = avctx->width  > 0 ? avctx->width  : 1920;
    int h = avctx->height > 0 ? avctx->height : 1080;

    /* OUTPUT: compressed bitstream */
    {
        struct v4l2_format fmt = {0};
        fmt.type                                 = OUT_TYPE;
        fmt.fmt.pix_mp.width                     = w;
        fmt.fmt.pix_mp.height                    = h;
        fmt.fmt.pix_mp.pixelformat               = s->v4l2_codec;
        fmt.fmt.pix_mp.field                     = V4L2_FIELD_NONE;
        fmt.fmt.pix_mp.num_planes                = 1;
        fmt.fmt.pix_mp.plane_fmt[0].sizeimage    = w * h;  /* bitstream buffer */
        if (xioctl(s->fd, VIDIOC_S_FMT, &fmt) < 0) {
            av_log(avctx, AV_LOG_ERROR,
                   "spacemit_mpp dec: S_FMT OUTPUT failed: %s\n", strerror(errno));
            return AVERROR(errno);
        }
        struct v4l2_format gfmt = {0};
        gfmt.type = OUT_TYPE;
        if (xioctl(s->fd, VIDIOC_G_FMT, &gfmt) == 0)
            s->out_num_planes = gfmt.fmt.pix_mp.num_planes;
        else
            s->out_num_planes = 1;
        av_log(avctx, AV_LOG_INFO,
               "spacemit_mpp dec: OUTPUT fmt=0x%x planes=%d\n",
               s->v4l2_codec, s->out_num_planes);
    }

    /* CAPTURE: decoded NV12 */
    {
        struct v4l2_format fmt = {0};
        fmt.type                                 = CAP_TYPE;
        fmt.fmt.pix_mp.width                     = w;
        fmt.fmt.pix_mp.height                    = h;
        fmt.fmt.pix_mp.pixelformat               = V4L2_PIX_FMT_NV12;
        fmt.fmt.pix_mp.field                     = V4L2_FIELD_NONE;
        fmt.fmt.pix_mp.num_planes                = 2;
        fmt.fmt.pix_mp.plane_fmt[0].sizeimage    = w * h;
        fmt.fmt.pix_mp.plane_fmt[1].sizeimage    = w * h / 2;
        if (xioctl(s->fd, VIDIOC_S_FMT, &fmt) < 0) {
            av_log(avctx, AV_LOG_ERROR,
                   "spacemit_mpp dec: S_FMT CAPTURE NV12 failed: %s\n", strerror(errno));
            return AVERROR(errno);
        }
        struct v4l2_format gfmt = {0};
        gfmt.type = CAP_TYPE;
        if (xioctl(s->fd, VIDIOC_G_FMT, &gfmt) == 0) {
            s->cap_num_planes = gfmt.fmt.pix_mp.num_planes;
            s->coded_width    = gfmt.fmt.pix_mp.width;
            s->coded_height   = gfmt.fmt.pix_mp.height;
        } else {
            s->cap_num_planes = 2;
            s->coded_width    = w;
            s->coded_height   = h;
        }
        av_log(avctx, AV_LOG_INFO,
               "spacemit_mpp dec: CAPTURE %dx%d planes=%d\n",
               s->coded_width, s->coded_height, s->cap_num_planes);
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Buffer allocation
 * ═══════════════════════════════════════════════════════════════════════════ */

static int alloc_buffers(AVCodecContext *avctx, __u32 type,
                          int n_req, int num_planes,
                          DecV4L2Buf **bufs_out, int *n_out)
{
    SpacemitMppDecContext *s = avctx->priv_data;
    struct v4l2_requestbuffers req = {0};
    req.count  = n_req;
    req.type   = type;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(s->fd, VIDIOC_REQBUFS, &req) < 0) {
        av_log(avctx, AV_LOG_ERROR,
               "spacemit_mpp dec: REQBUFS type=%u failed: %s\n",
               type, strerror(errno));
        return AVERROR(errno);
    }

    *n_out  = req.count;
    *bufs_out = av_calloc(req.count, sizeof(DecV4L2Buf));
    if (!*bufs_out) return AVERROR(ENOMEM);

    for (unsigned i = 0; i < req.count; i++) {
        struct v4l2_buffer     buf  = {0};
        struct v4l2_plane      planes[SPACEMIT_DEC_MAX_PLANES] = {{0}};
        buf.index    = i;
        buf.type     = type;
        buf.memory   = V4L2_MEMORY_MMAP;
        buf.length   = num_planes;
        buf.m.planes = planes;

        if (xioctl(s->fd, VIDIOC_QUERYBUF, &buf) < 0) {
            av_log(avctx, AV_LOG_ERROR,
                   "spacemit_mpp dec: QUERYBUF %u failed: %s\n", i, strerror(errno));
            return AVERROR(errno);
        }

        (*bufs_out)[i].num_planes = num_planes;
        (*bufs_out)[i].index      = i;
        for (int p = 0; p < num_planes; p++) {
            (*bufs_out)[i].length[p] = planes[p].length;
            (*bufs_out)[i].start[p]  = mmap(NULL, planes[p].length,
                                             PROT_READ | PROT_WRITE, MAP_SHARED,
                                             s->fd, planes[p].m.mem_offset);
            if ((*bufs_out)[i].start[p] == MAP_FAILED) {
                av_log(avctx, AV_LOG_ERROR,
                       "spacemit_mpp dec: mmap buf %u plane %d failed: %s\n",
                       i, p, strerror(errno));
                return AVERROR(errno);
            }
        }
    }
    return 0;
}

static void free_dec_buffers(DecV4L2Buf *bufs, int n)
{
    if (!bufs) return;
    for (int i = 0; i < n; i++)
        for (int p = 0; p < bufs[i].num_planes; p++)
            if (bufs[i].start[p] && bufs[i].start[p] != MAP_FAILED)
                munmap(bufs[i].start[p], bufs[i].length[p]);
    av_free(bufs);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Queue all CAPTURE buffers so the driver can fill them
 * ═══════════════════════════════════════════════════════════════════════════ */

static int queue_all_capture_bufs(AVCodecContext *avctx)
{
    SpacemitMppDecContext *s = avctx->priv_data;
    for (int i = 0; i < s->n_cap; i++) {
        struct v4l2_buffer  buf    = {0};
        struct v4l2_plane   planes[SPACEMIT_DEC_MAX_PLANES] = {{0}};
        buf.index    = i;
        buf.type     = CAP_TYPE;
        buf.memory   = V4L2_MEMORY_MMAP;
        buf.length   = s->cap_num_planes;
        buf.m.planes = planes;
        for (int p = 0; p < s->cap_num_planes; p++)
            planes[p].length = s->cap_bufs[i].length[p];
        if (xioctl(s->fd, VIDIOC_QBUF, &buf) < 0) {
            av_log(avctx, AV_LOG_ERROR,
                   "spacemit_mpp dec: QBUF CAPTURE %d failed: %s\n",
                   i, strerror(errno));
            return AVERROR(errno);
        }
        s->cap_bufs[i].queued = 1;
    }
    return 0;
}

static int stream_on(AVCodecContext *avctx, __u32 type)
{
    SpacemitMppDecContext *s = avctx->priv_data;
    if (xioctl(s->fd, VIDIOC_STREAMON, &type) < 0) {
        av_log(avctx, AV_LOG_ERROR,
               "spacemit_mpp dec: STREAMON type=%u failed: %s\n",
               type, strerror(errno));
        return AVERROR(errno);
    }
    return 0;
}

static void stream_off_dec(SpacemitMppDecContext *s)
{
    __u32 t;
    t = OUT_TYPE; xioctl(s->fd, VIDIOC_STREAMOFF, &t);
    t = CAP_TYPE; xioctl(s->fd, VIDIOC_STREAMOFF, &t);
    s->output_started  = 0;
    s->capture_started = 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Init / Close
 * ═══════════════════════════════════════════════════════════════════════════ */

static av_cold int spacemit_mpp_decode_init(AVCodecContext *avctx)
{
    SpacemitMppDecContext *s = avctx->priv_data;
    int ret;

    s->fd          = -1;
    s->v4l2_codec  = avcodec_to_v4l2_codec(avctx->codec_id);
    s->num_out_bufs = s->num_out_bufs > 0 ? s->num_out_bufs : SPACEMIT_DEC_OUTPUT_BUFS;
    s->num_cap_bufs = s->num_cap_bufs > 0 ? s->num_cap_bufs : SPACEMIT_DEC_CAPTURE_BUFS;

    if (!s->v4l2_codec) {
        av_log(avctx, AV_LOG_ERROR, "spacemit_mpp: unsupported codec %d\n",
               avctx->codec_id);
        return AVERROR(ENOSYS);
    }

    /* Initialize pts queue */
    for (int i = 0; i < (int)FF_ARRAY_ELEMS(s->pts_queue); i++)
        s->pts_queue[i] = AV_NOPTS_VALUE;

    if ((ret = find_v4l2_device(avctx, s->v4l2_codec)) < 0)
        return ret;

    if ((ret = set_formats_mplane(avctx)) < 0)
        goto fail;

    if ((ret = alloc_buffers(avctx, OUT_TYPE, s->num_out_bufs,
                              s->out_num_planes, &s->out_bufs, &s->n_out)) < 0)
        goto fail;

    if ((ret = alloc_buffers(avctx, CAP_TYPE, s->num_cap_bufs,
                              s->cap_num_planes, &s->cap_bufs, &s->n_cap)) < 0)
        goto fail;

    if ((ret = queue_all_capture_bufs(avctx)) < 0)
        goto fail;

    if ((ret = stream_on(avctx, CAP_TYPE)) < 0)
        goto fail;
    s->capture_started = 1;

    avctx->pix_fmt = AV_PIX_FMT_NV12;
    if (avctx->width  <= 0) avctx->width  = s->coded_width;
    if (avctx->height <= 0) avctx->height = s->coded_height;

    return 0;

fail:
    free_dec_buffers(s->out_bufs, s->n_out); s->out_bufs = NULL;
    free_dec_buffers(s->cap_bufs, s->n_cap); s->cap_bufs = NULL;
    if (s->fd >= 0) { close(s->fd); s->fd = -1; }
    return ret;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Send packet — copy compressed data into an OUTPUT buffer and queue it
 * ═══════════════════════════════════════════════════════════════════════════ */

static int spacemit_mpp_send_packet(AVCodecContext *avctx, const AVPacket *pkt)
{
    SpacemitMppDecContext *s = avctx->priv_data;
    struct v4l2_buffer  buf    = {0};
    struct v4l2_plane   planes[SPACEMIT_DEC_MAX_PLANES] = {{0}};
    int i;

    /* EOS: send zero-length packet */
    if (!pkt || pkt->size == 0) {
        buf.type     = OUT_TYPE;
        buf.memory   = V4L2_MEMORY_MMAP;
        buf.length   = s->out_num_planes;
        buf.m.planes = planes;
        buf.flags    = V4L2_BUF_FLAG_LAST;
        /* find any free output buffer */
        for (i = 0; i < s->n_out; i++)
            if (!s->out_bufs[i].queued) break;
        if (i == s->n_out) return AVERROR(EAGAIN);
        buf.index = i;
        planes[0].bytesused = 0;
        planes[0].length    = s->out_bufs[i].length[0];
        s->out_bufs[i].queued = 1;
        s->draining = 1;
        if (xioctl(s->fd, VIDIOC_QBUF, &buf) < 0)
            return AVERROR(errno);
        return 0;
    }

    if (pkt->size > (int)s->out_bufs[0].length[0]) {
        av_log(avctx, AV_LOG_ERROR,
               "spacemit_mpp dec: packet size %d > buffer %zu\n",
               pkt->size, s->out_bufs[0].length[0]);
        return AVERROR(EINVAL);
    }

    /* Find a free OUTPUT buffer */
    for (i = 0; i < s->n_out; i++)
        if (!s->out_bufs[i].queued) break;
    if (i == s->n_out) return AVERROR(EAGAIN);

    memcpy(s->out_bufs[i].start[0], pkt->data, pkt->size);
    s->pts_queue[i % (int)FF_ARRAY_ELEMS(s->pts_queue)] = pkt->pts;

    buf.index    = i;
    buf.type     = OUT_TYPE;
    buf.memory   = V4L2_MEMORY_MMAP;
    buf.length   = s->out_num_planes;
    buf.m.planes = planes;
    buf.timestamp.tv_sec  = pkt->pts / AV_TIME_BASE;
    buf.timestamp.tv_usec = (pkt->pts % AV_TIME_BASE) * 1000000 / AV_TIME_BASE;
    planes[0].bytesused = pkt->size;
    planes[0].length    = s->out_bufs[i].length[0];

    s->out_bufs[i].queued = 1;
    if (xioctl(s->fd, VIDIOC_QBUF, &buf) < 0) {
        s->out_bufs[i].queued = 0;
        av_log(avctx, AV_LOG_ERROR,
               "spacemit_mpp dec: QBUF OUTPUT %d failed: %s\n", i, strerror(errno));
        return AVERROR(errno);
    }

    if (!s->output_started) {
        if (stream_on(avctx, OUT_TYPE) < 0)
            return AVERROR(errno);
        s->output_started = 1;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Receive frame — dequeue a CAPTURE buffer and copy into AVFrame
 * ═══════════════════════════════════════════════════════════════════════════ */

static int spacemit_mpp_receive_frame(AVCodecContext *avctx, AVFrame *frame)
{
    SpacemitMppDecContext *s = avctx->priv_data;
    struct v4l2_buffer  cap    = {0};
    struct v4l2_plane   cplanes[SPACEMIT_DEC_MAX_PLANES] = {{0}};
    struct v4l2_buffer  out    = {0};
    struct v4l2_plane   oplanes[SPACEMIT_DEC_MAX_PLANES] = {{0}};
    struct pollfd       pfd    = { .fd = s->fd, .events = POLLIN | POLLOUT };
    int ret;

    if (s->eof) return AVERROR_EOF;

    /* Poll for data availability (100 ms timeout) */
    ret = poll(&pfd, 1, 100);
    if (ret < 0)  return AVERROR(errno);
    if (ret == 0) return AVERROR(EAGAIN);  /* timeout */

    /* Reclaim any done OUTPUT buffers */
    if (pfd.revents & POLLOUT) {
        out.type     = OUT_TYPE;
        out.memory   = V4L2_MEMORY_MMAP;
        out.length   = s->out_num_planes;
        out.m.planes = oplanes;
        while (xioctl(s->fd, VIDIOC_DQBUF, &out) == 0) {
            if (out.index < (unsigned)s->n_out)
                s->out_bufs[out.index].queued = 0;
            /* Reset for next call */
            memset(&out,    0, sizeof(out));
            memset(oplanes, 0, sizeof(oplanes));
            out.type     = OUT_TYPE;
            out.memory   = V4L2_MEMORY_MMAP;
            out.length   = s->out_num_planes;
            out.m.planes = oplanes;
        }
    }

    /* Dequeue a decoded CAPTURE frame */
    if (!(pfd.revents & POLLIN))
        return AVERROR(EAGAIN);

    cap.type     = CAP_TYPE;
    cap.memory   = V4L2_MEMORY_MMAP;
    cap.length   = s->cap_num_planes;
    cap.m.planes = cplanes;
    if (xioctl(s->fd, VIDIOC_DQBUF, &cap) < 0) {
        if (errno == EPIPE || errno == EAGAIN)
            return s->draining ? AVERROR_EOF : AVERROR(EAGAIN);
        av_log(avctx, AV_LOG_ERROR,
               "spacemit_mpp dec: DQBUF CAPTURE failed: %s\n", strerror(errno));
        return AVERROR(errno);
    }

    /* EOS marker from the driver */
    if (cap.flags & V4L2_BUF_FLAG_LAST) {
        /* Re-queue the buffer even if empty */
        xioctl(s->fd, VIDIOC_QBUF, &cap);
        s->eof = 1;
        return AVERROR_EOF;
    }

    /* Build AVFrame from the decoded NV12 buffer */
    frame->format = AV_PIX_FMT_NV12;
    frame->width  = s->coded_width;
    frame->height = s->coded_height;
    if ((ret = av_frame_get_buffer(frame, 32)) < 0) {
        xioctl(s->fd, VIDIOC_QBUF, &cap);
        return ret;
    }

    /* Y plane */
    int y_size   = s->coded_width * s->coded_height;
    int uv_size  = y_size / 2;
    int buf_idx  = cap.index;

    if (s->cap_num_planes >= 2) {
        /* Separate Y and UV planes (driver gave us 2 planes) */
        memcpy(frame->data[0], s->cap_bufs[buf_idx].start[0], y_size);
        memcpy(frame->data[1], s->cap_bufs[buf_idx].start[1], uv_size);
    } else {
        /* Contiguous NV12 (Y immediately followed by UV) */
        memcpy(frame->data[0], s->cap_bufs[buf_idx].start[0], y_size);
        memcpy(frame->data[1],
               (uint8_t *)s->cap_bufs[buf_idx].start[0] + y_size,
               uv_size);
    }

    /* Restore PTS */
    frame->pts = s->pts_queue[buf_idx % (int)FF_ARRAY_ELEMS(s->pts_queue)];

    /* Re-queue the CAPTURE buffer */
    if (xioctl(s->fd, VIDIOC_QBUF, &cap) < 0)
        av_log(avctx, AV_LOG_WARNING,
               "spacemit_mpp dec: QBUF CAPTURE %u failed: %s\n",
               cap.index, strerror(errno));

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Flush — drain without destroying the decoder state
 * ═══════════════════════════════════════════════════════════════════════════ */

static void spacemit_mpp_decode_flush(AVCodecContext *avctx)
{
    SpacemitMppDecContext *s = avctx->priv_data;
    __u32 t;
    t = OUT_TYPE; xioctl(s->fd, VIDIOC_STREAMOFF, &t);
    t = CAP_TYPE; xioctl(s->fd, VIDIOC_STREAMOFF, &t);
    s->output_started  = 0;
    s->capture_started = 0;
    s->draining        = 0;
    s->eof             = 0;
    /* Re-queue all capture buffers */
    queue_all_capture_bufs(avctx);
    stream_on(avctx, CAP_TYPE);
    s->capture_started = 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Close
 * ═══════════════════════════════════════════════════════════════════════════ */

static av_cold int spacemit_mpp_decode_close(AVCodecContext *avctx)
{
    SpacemitMppDecContext *s = avctx->priv_data;
    stream_off_dec(s);
    free_dec_buffers(s->out_bufs, s->n_out); s->out_bufs = NULL;
    free_dec_buffers(s->cap_bufs, s->n_cap); s->cap_bufs = NULL;
    if (s->fd >= 0) { close(s->fd); s->fd = -1; }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Options, pixel formats, FFCodec registration
 * ═══════════════════════════════════════════════════════════════════════════ */

static const AVOption spacemit_mpp_dec_options[] = {
    { "device",
      "V4L2 device node (empty = auto-detect)",
      offsetof(SpacemitMppDecContext, device),
      AV_OPT_TYPE_STRING, { .str = "" }, 0, 0,
      AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM },
    { "num_output_bufs",
      "number of V4L2 OUTPUT (bitstream) queue buffers",
      offsetof(SpacemitMppDecContext, num_out_bufs),
      AV_OPT_TYPE_INT, { .i64 = SPACEMIT_DEC_OUTPUT_BUFS }, 2, 32,
      AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM },
    { "num_capture_bufs",
      "number of V4L2 CAPTURE (frame) queue buffers",
      offsetof(SpacemitMppDecContext, num_cap_bufs),
      AV_OPT_TYPE_INT, { .i64 = SPACEMIT_DEC_CAPTURE_BUFS }, 2, 32,
      AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM },
    { NULL }
};

#define DEFINE_SPACEMIT_MPP_DECODER(SHORT, FULLNAME, AVID)                    \
static const AVClass spacemit_mpp_dec_##SHORT##_class = {                     \
    .class_name = #SHORT "_spacemit_mpp",                                     \
    .item_name  = av_default_item_name,                                       \
    .option     = spacemit_mpp_dec_options,                                   \
    .version    = LIBAVUTIL_VERSION_INT,                                      \
};                                                                             \
const FFCodec ff_##SHORT##_spacemit_mpp_decoder = {                           \
    .p.name         = #SHORT "_spacemit_mpp",                                 \
    CODEC_LONG_NAME(FULLNAME " (SpacemiT K1 MPP, V4L2 M2M)"),                \
    .p.type         = AVMEDIA_TYPE_VIDEO,                                     \
    .p.id           = AVID,                                                   \
    .priv_data_size = sizeof(SpacemitMppDecContext),                          \
    .p.priv_class   = &spacemit_mpp_dec_##SHORT##_class,                     \
    .init           = spacemit_mpp_decode_init,                               \
    FF_CODEC_RECEIVE_FRAME_CB(spacemit_mpp_receive_frame),                    \
    .flush          = spacemit_mpp_decode_flush,                              \
    .close          = spacemit_mpp_decode_close,                              \
    .p.capabilities = AV_CODEC_CAP_HARDWARE | AV_CODEC_CAP_DR1               \
                    | AV_CODEC_CAP_AVOID_PROBING,                             \
    .p.wrapper_name = "spacemit_mpp",                                         \
    .caps_internal  = FF_CODEC_CAP_NOT_INIT_THREADSAFE,                      \
};

DEFINE_SPACEMIT_MPP_DECODER(h264, "H.264 / AVC",  AV_CODEC_ID_H264)
DEFINE_SPACEMIT_MPP_DECODER(hevc, "H.265 / HEVC", AV_CODEC_ID_HEVC)
DEFINE_SPACEMIT_MPP_DECODER(vp8,  "VP8",           AV_CODEC_ID_VP8)
DEFINE_SPACEMIT_MPP_DECODER(vp9,  "VP9",           AV_CODEC_ID_VP9)
