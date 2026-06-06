/*
 * vf_scale_spacemit.c — SpacemiT K1 V2D hardware scale filter for FFmpeg
 *
 * Transparently activates the V2D 2D-blitter inside any filter graph.
 * Drop-in replacement for the software `scale` filter on K1 platforms:
 *
 *   ffmpeg -i in.mp4 -vf scale_spacemit=1280:720 out.mp4
 *   ffmpeg -i in.mp4 -vf scale_spacemit=w=1280:h=720 out.mp4
 *
 * Design
 * ──────
 *   Input      : AV_PIX_FMT_NV12 or AV_PIX_FMT_YUV420P (software frames)
 *   Output     : AV_PIX_FMT_NV12 at the requested dimensions
 *   DMA heap   : /dev/dma_heap/system (no extra library required)
 *   V2D        : V2D_AddBlendTask with V2D_COLOR_FORMAT_NV12; the blitter
 *                automatically applies bilinear scaling when src≠dst size.
 *   Fallback   : silent libswscale CPU path when V2D is unavailable.
 *
 * V2D library : /usr/lib/libv2d.so  (link with -lv2d)
 * V2D headers : /usr/include/spacemit/v2d_api.h  v2d_type.h  v2d_compat.h
 *
 * This file is part of FFmpeg / ffmpeg-spacemit.
 */

/* Include glibc POSIX headers before any Linux kernel headers to ensure
 * O_RDWR, O_CLOEXEC, PROT_*, MAP_* are defined from glibc, not the kernel. */
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

/* O_CLOEXEC: defined by glibc ≥ 2.9 / Linux ≥ 2.6.23.
 * Add a fallback in case an unusual toolchain doesn't expose it. */
#ifndef O_CLOEXEC
#  define O_CLOEXEC 02000000
#endif

/* DMA-heap kernel interface (uses __u64/__u32, no POSIX symbols) */
#include <linux/dma-heap.h>

/* v2d_compat.h maps V2D_BeginJob → ASR_V2D_BeginJob etc.; must precede api.h */
#include <spacemit/v2d_compat.h>
#include <spacemit/v2d_type.h>
#include <spacemit/v2d_api.h>

#include "libavutil/avassert.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libswscale/swscale.h"
#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "scale_eval.h"
#include "video.h"

/* ── DMA-heap buffer ───────────────────────────────────────────────────── */

#define ALIGN4K(s)  (((size_t)(s) + 4095UL) & ~4095UL)

typedef struct DMABuf {
    int    fd;
    void  *ptr;
    size_t size;
} DMABuf;

static void dmabuf_free(DMABuf *b)
{
    if (!b) return;
    if (b->ptr && b->ptr != MAP_FAILED) {
        munmap(b->ptr, b->size);
        b->ptr = NULL;
    }
    if (b->fd >= 0) {
        close(b->fd);
        b->fd = -1;
    }
    b->size = 0;
}

/**
 * Allocate a contiguous DMA-heap buffer accessible by V2D.
 * Returns 0 on success, negative on error.
 */
static int dmabuf_alloc(DMABuf *b, size_t size)
{
    dmabuf_free(b);

    int heap_fd = open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
    if (heap_fd < 0)
        return AVERROR(errno);

    struct dma_heap_allocation_data req = {
        .len        = ALIGN4K(size),
        .fd_flags   = O_RDWR | O_CLOEXEC,
        .heap_flags = 0,
    };

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &req) < 0) {
        int err = errno;
        close(heap_fd);
        return AVERROR(err);
    }
    close(heap_fd);

    void *ptr = mmap(NULL, ALIGN4K(size), PROT_READ | PROT_WRITE,
                     MAP_SHARED, req.fd, 0);
    if (ptr == MAP_FAILED) {
        close(req.fd);
        return AVERROR(ENOMEM);
    }

    b->fd   = (int)req.fd;
    b->ptr  = ptr;
    b->size = ALIGN4K(size);
    return 0;
}

/* ── Filter context ────────────────────────────────────────────────────── */

typedef struct ScaleSpacemitContext {
    const AVClass *class;

    /* User options */
    char *w_expr;
    char *h_expr;
    int   force_original_aspect_ratio;
    int   force_divisible_by;

    /* Resolved output dimensions (set during config_output) */
    int out_w, out_h;

    /* Hardware availability flag (0 = CPU fallback) */
    int v2d_ok;

    /* Pre-allocated DMA buffers; reallocated only on dimension change */
    DMABuf src_dma;       /* NV12 input,  src_w × src_h    */
    DMABuf dst_dma;       /* NV12 output, out_w × out_h    */
    int    src_w, src_h;  /* current src_dma frame size     */

    /* libswscale context — used as the full CPU fallback path */
    struct SwsContext *sws;

    /* Chosen output pixel format (NV12 or YUV420P), picked by
     * filter-graph negotiation in scale_spacemit_config_output. */
    enum AVPixelFormat out_format;
} ScaleSpacemitContext;

/* ── V2D scale via NV12-native BlendTask ───────────────────────────────── */

/**
 * Scale src_fd (sw×sh NV12) → dst_fd (dw×dh NV12) via V2D hardware.
 *
 * Both buffers must be contiguous DMA-heap allocations with UV immediately
 * following Y (offset = stride × height).  The V2D blitter applies bilinear
 * scaling automatically when source and destination rectangles differ.
 */
static int v2d_scale_nv12(AVFilterContext *avctx,
                           int src_fd, int sw, int sh,
                           int dst_fd, int dw, int dh)
{
    V2D_SURFACE_S    bg      = {0};
    V2D_SURFACE_S    dst_s   = {0};
    V2D_AREA_S       bg_r    = {0};
    V2D_AREA_S       dst_r   = {0};
    V2D_BLEND_CONF_S blend   = {0};

    /* Source surface (background layer = layer 0) */
    bg.fbc_enable  = 0;
    bg.fd          = src_fd;
    bg.offset      = (uint32_t)(sw * sh);   /* UV immediately after Y */
    bg.w           = (uint16_t)sw;
    bg.h           = (uint16_t)sh;
    bg.stride      = (uint16_t)sw;
    bg.format      = V2D_COLOR_FORMAT_NV12;
    bg_r.x = 0; bg_r.y = 0;
    bg_r.w = (uint16_t)sw; bg_r.h = (uint16_t)sh;

    /* Destination surface */
    dst_s.fbc_enable  = 0;
    dst_s.fd          = dst_fd;
    dst_s.offset      = (uint32_t)(dw * dh);
    dst_s.w           = (uint16_t)dw;
    dst_s.h           = (uint16_t)dh;
    dst_s.stride      = (uint16_t)dw;
    dst_s.format      = V2D_COLOR_FORMAT_NV12;
    dst_r.x = 0; dst_r.y = 0;
    dst_r.w = (uint16_t)dw; dst_r.h = (uint16_t)dh;

    /* Blend config: pass-through (background only, no foreground, alpha=1) */
    blend.blend_cmd = V2D_BLENDCMD_ALPHA;
    blend.blendlayer[0].blend_area.x = 0;
    blend.blendlayer[0].blend_area.y = 0;
    blend.blendlayer[0].blend_area.w = (uint16_t)dw;
    blend.blendlayer[0].blend_area.h = (uint16_t)dh;

    V2D_HANDLE job;
    if (V2D_BeginJob(&job) != SUCCESS) {
        av_log(avctx, AV_LOG_WARNING, "scale_spacemit: V2D_BeginJob failed\n");
        return AVERROR_EXTERNAL;
    }

    int r = V2D_AddBlendTask(job,
                             &bg,    &bg_r,   /* background (source)  */
                             NULL,   NULL,    /* foreground (none)    */
                             NULL,   NULL,    /* mask (none)          */
                             &dst_s, &dst_r,
                             &blend,
                             V2D_ROT_0, V2D_ROT_0,
                             V2D_CSC_MODE_RGB_2_RGB,  /* NV12→NV12 same space */
                             V2D_CSC_MODE_BUTT,
                             NULL, V2D_NO_DITHER);
    if (r != SUCCESS) {
        V2D_EndJob(job);
        av_log(avctx, AV_LOG_WARNING,
               "scale_spacemit: V2D_AddBlendTask failed (%d)\n", r);
        return AVERROR_EXTERNAL;
    }

    r = V2D_EndJob(job);
    if (r != SUCCESS) {
        av_log(avctx, AV_LOG_WARNING,
               "scale_spacemit: V2D_EndJob failed (%d)\n", r);
        return AVERROR_EXTERNAL;
    }
    return 0;
}

/* ── Stride-aware NV12 frame ↔ contiguous DMA buffer ──────────────────── */

static void frame_to_dma_nv12(const AVFrame *f, uint8_t *dma, int w, int h)
{
    const uint8_t *src;
    uint8_t       *dst;

    /* Y plane: copy line by line to strip linesize padding */
    src = f->data[0];
    dst = dma;
    for (int y = 0; y < h; y++) {
        memcpy(dst, src, w);
        src += f->linesize[0];
        dst += w;
    }
    /* UV plane: interleaved, h/2 rows */
    src = f->data[1];
    dst = dma + (size_t)w * h;
    for (int y = 0; y < h / 2; y++) {
        memcpy(dst, src, w);
        src += f->linesize[1];
        dst += w;
    }
}

static void dma_to_frame_nv12(const uint8_t *dma, AVFrame *f, int w, int h)
{
    const uint8_t *src;
    uint8_t       *dst;

    src = dma;
    dst = f->data[0];
    for (int y = 0; y < h; y++) {
        memcpy(dst, src, w);
        src += w;
        dst += f->linesize[0];
    }
    src = dma + (size_t)w * h;
    dst = f->data[1];
    for (int y = 0; y < h / 2; y++) {
        memcpy(dst, src, w);
        src += w;
        dst += f->linesize[1];
    }
}

/* ── Stride-aware YUV420P frame ↔ contiguous DMA buffer ──────────────────── */

static void frame_to_dma_yuv420p(const AVFrame *f, uint8_t *dma, int w, int h)
{
    const uint8_t *src;
    uint8_t       *dst;
    int            cw = w / 2, ch = h / 2;

    /* Y plane */
    src = f->data[0];
    dst = dma;
    for (int y = 0; y < h; y++) { memcpy(dst, src, w); src += f->linesize[0]; dst += w; }
    /* U plane */
    src = f->data[1];
    dst = dma + (size_t)w * h;
    for (int y = 0; y < ch; y++) { memcpy(dst, src, cw); src += f->linesize[1]; dst += cw; }
    /* V plane */
    src = f->data[2];
    dst = dma + (size_t)w * h + (size_t)cw * ch;
    for (int y = 0; y < ch; y++) { memcpy(dst, src, cw); src += f->linesize[2]; dst += cw; }
}

static void dma_to_frame_yuv420p(const uint8_t *dma, AVFrame *f, int w, int h)
{
    const uint8_t *src;
    uint8_t       *dst;
    int            cw = w / 2, ch = h / 2;

    src = dma;                 dst = f->data[0];
    for (int y = 0; y < h;  y++) { memcpy(dst, src, w);  src += w;  dst += f->linesize[0]; }
    src = dma + (size_t)w * h; dst = f->data[1];
    for (int y = 0; y < ch; y++) { memcpy(dst, src, cw); src += cw; dst += f->linesize[1]; }
    src = dma + (size_t)w * h + (size_t)cw * ch; dst = f->data[2];
    for (int y = 0; y < ch; y++) { memcpy(dst, src, cw); src += cw; dst += f->linesize[2]; }
}

/* ── V2D scale via three Y8 bit-blits (Y, U, V) — YUV420P-native ─────────
 *
 * The V2D library has no YUV420P color format constant, but YUV420P is just
 * three independent Y8 (single-byte-per-pixel) planes: Y is w×h, U and V
 * are each w/2 × h/2.  V2D_AddBitblitTask applies bilinear scaling per
 * surface, so three Y8 bit-blits give us a full YUV420P scale with the
 * pipeline staying in YUV420P throughout — no NV12 conversion anywhere.
 */
static int v2d_blit_y8(V2D_HANDLE job,
                       int src_fd, uint32_t src_off, int sw, int sh,
                       int dst_fd, uint32_t dst_off, int dw, int dh)
{
    V2D_SURFACE_S s = {0}, d = {0};
    V2D_AREA_S    sa = {0}, da = {0};

    s.fd     = src_fd; s.offset = src_off;
    s.w      = (uint16_t)sw; s.h = (uint16_t)sh; s.stride = (uint16_t)sw;
    s.format = V2D_COLOR_FORMAT_Y8;
    d.fd     = dst_fd; d.offset = dst_off;
    d.w      = (uint16_t)dw; d.h = (uint16_t)dh; d.stride = (uint16_t)dw;
    d.format = V2D_COLOR_FORMAT_Y8;
    sa.w = (uint16_t)sw; sa.h = (uint16_t)sh;
    da.w = (uint16_t)dw; da.h = (uint16_t)dh;

    return V2D_AddBitblitTask(job, &d, &da, &s, &sa, V2D_CSC_MODE_BUTT);
}

static int v2d_scale_yuv420p(AVFilterContext *avctx,
                              int src_fd, int sw, int sh,
                              int dst_fd, int dw, int dh)
{
    int scw = sw / 2, sch = sh / 2;
    int dcw = dw / 2, dch = dh / 2;
    uint32_t s_u_off = (uint32_t)sw * sh;
    uint32_t s_v_off = s_u_off + (uint32_t)scw * sch;
    uint32_t d_u_off = (uint32_t)dw * dh;
    uint32_t d_v_off = d_u_off + (uint32_t)dcw * dch;

    V2D_HANDLE job;
    if (V2D_BeginJob(&job) != SUCCESS) {
        av_log(avctx, AV_LOG_WARNING,
               "scale_spacemit: V2D_BeginJob failed (yuv420p)\n");
        return AVERROR_EXTERNAL;
    }
    if (v2d_blit_y8(job, src_fd, 0,        sw,  sh,  dst_fd, 0,        dw,  dh) != SUCCESS ||
        v2d_blit_y8(job, src_fd, s_u_off,  scw, sch, dst_fd, d_u_off,  dcw, dch) != SUCCESS ||
        v2d_blit_y8(job, src_fd, s_v_off,  scw, sch, dst_fd, d_v_off,  dcw, dch) != SUCCESS) {
        V2D_EndJob(job);
        av_log(avctx, AV_LOG_WARNING,
               "scale_spacemit: V2D_AddBitblitTask failed (yuv420p)\n");
        return AVERROR_EXTERNAL;
    }
    if (V2D_EndJob(job) != SUCCESS) {
        av_log(avctx, AV_LOG_WARNING,
               "scale_spacemit: V2D_EndJob failed (yuv420p)\n");
        return AVERROR_EXTERNAL;
    }
    return 0;
}

/* ── AVFilter callbacks ────────────────────────────────────────────────── */

static av_cold int scale_spacemit_init(AVFilterContext *avctx)
{
    ScaleSpacemitContext *s = avctx->priv;

    s->src_dma.fd = -1;
    s->dst_dma.fd = -1;

    /* Probe: can we open the V2D device? */
    int fd = open("/dev/v2d_dev", O_RDWR | O_CLOEXEC);
    if (fd >= 0) {
        close(fd);
        s->v2d_ok = 1;
        av_log(avctx, AV_LOG_INFO,
               "scale_spacemit: V2D hardware available\n");
    } else {
        s->v2d_ok = 0;
        av_log(avctx, AV_LOG_WARNING,
               "scale_spacemit: /dev/v2d_dev unavailable (%s) — "
               "falling back to libswscale\n", strerror(errno));
    }
    return 0;
}

static av_cold void scale_spacemit_uninit(AVFilterContext *avctx)
{
    ScaleSpacemitContext *s = avctx->priv;
    dmabuf_free(&s->src_dma);
    dmabuf_free(&s->dst_dma);
    sws_freeContext(s->sws);
    s->sws = NULL;
}

static int scale_spacemit_query_formats(const AVFilterContext *ctx,
                                         AVFilterFormatsConfig **cfg_in,
                                         AVFilterFormatsConfig **cfg_out)
{
    /* Input: accept NV12 and YUV420P (the most common decoded formats) */
    static const enum AVPixelFormat in_fmts[] = {
        AV_PIX_FMT_NV12, AV_PIX_FMT_YUV420P, AV_PIX_FMT_NONE
    };
    /* Output: NV12 (single V2D NV12-native blit) or YUV420P (three Y8
     * blits, no NV12 conversion anywhere).  Filter-graph negotiation
     * picks whichever the downstream prefers. */
    static const enum AVPixelFormat out_fmts[] = {
        AV_PIX_FMT_NV12, AV_PIX_FMT_YUV420P, AV_PIX_FMT_NONE
    };

    int ret;
    ret = ff_formats_ref(ff_make_format_list(in_fmts),  &cfg_in[0]->formats);
    if (ret < 0) return ret;
    ret = ff_formats_ref(ff_make_format_list(out_fmts), &cfg_out[0]->formats);
    return ret;
}

static int scale_spacemit_config_output(AVFilterLink *outlink)
{
    AVFilterContext      *avctx  = outlink->src;
    AVFilterLink         *inlink = avctx->inputs[0];
    ScaleSpacemitContext *s      = avctx->priv;
    int ret;

    /* Resolve w/h expressions (supports iw, ih, dar, sar, etc.) */
    ret = ff_scale_eval_dimensions(s,
                                   s->w_expr, s->h_expr,
                                   inlink, outlink,
                                   &s->out_w, &s->out_h);
    if (ret < 0)
        return ret;

    ff_scale_adjust_dimensions(inlink,
                               &s->out_w, &s->out_h,
                               s->force_original_aspect_ratio,
                               s->force_divisible_by,
                               1.0);

    /* V2D NV12 requires even dimensions */
    s->out_w = FFALIGN(s->out_w, 2);
    s->out_h = FFALIGN(s->out_h, 2);

    outlink->w      = s->out_w;
    outlink->h      = s->out_h;
    /* Filter graph negotiation has already chosen outlink->format from
     * the candidates we advertised; default to NV12 if unset. */
    if (outlink->format == AV_PIX_FMT_NONE)
        outlink->format = AV_PIX_FMT_NV12;
    s->out_format = outlink->format;

    /* Propagate SAR, adjusting for the new AR */
    if (inlink->sample_aspect_ratio.num) {
        outlink->sample_aspect_ratio =
            av_mul_q((AVRational){ outlink->h * inlink->w,
                                   outlink->w * inlink->h },
                     inlink->sample_aspect_ratio);
    } else {
        outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;
    }

    av_log(avctx, AV_LOG_VERBOSE,
           "scale_spacemit: %dx%d %s → %dx%d %s  [%s]\n",
           inlink->w, inlink->h,
           av_get_pix_fmt_name(inlink->format),
           s->out_w, s->out_h,
           av_get_pix_fmt_name(s->out_format),
           s->v2d_ok ? "V2D HW" : "swscale CPU");

    /* Pre-allocate destination DMA buffer (size is fixed at output dims) */
    if (s->v2d_ok) {
        size_t dst_sz = (size_t)s->out_w * s->out_h * 3 / 2;
        ret = dmabuf_alloc(&s->dst_dma, dst_sz);
        if (ret < 0) {
            av_log(avctx, AV_LOG_WARNING,
                   "scale_spacemit: dst DMA alloc failed (%s) — "
                   "disabling V2D\n", av_err2str(ret));
            s->v2d_ok = 0;
        }
    }
    return 0;
}

static int scale_spacemit_filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext      *avctx   = inlink->dst;
    AVFilterLink         *outlink = avctx->outputs[0];
    ScaleSpacemitContext *s       = avctx->priv;
    AVFrame *out = NULL;
    int ret;

    /* ── Passthrough: same dims AND already the format the downstream wants ── */
    if (in->width == s->out_w && in->height == s->out_h &&
        in->format == s->out_format) {
        return ff_filter_frame(outlink, in);
    }

    /* ── V2D hardware path ── */
    if (s->v2d_ok) {
        int in_w = in->width;
        int in_h = in->height;

        /* Reallocate src DMA buffer if input size changed */
        if (in_w != s->src_w || in_h != s->src_h || s->src_dma.fd < 0) {
            dmabuf_free(&s->src_dma);
            size_t src_sz = (size_t)in_w * in_h * 3 / 2;
            ret = dmabuf_alloc(&s->src_dma, src_sz);
            if (ret < 0) {
                av_log(avctx, AV_LOG_WARNING,
                       "scale_spacemit: src DMA alloc failed (%s) — "
                       "disabling V2D\n", av_err2str(ret));
                s->v2d_ok = 0;
                goto sw_fallback;
            }
            s->src_w = in_w;
            s->src_h = in_h;
        }

        /* Stage input into contiguous DMA buffer in whichever layout
         * V2D will use to scale (matches in->format). */
        if (in->format == AV_PIX_FMT_NV12)
            frame_to_dma_nv12(in, s->src_dma.ptr, in_w, in_h);
        else  /* AV_PIX_FMT_YUV420P */
            frame_to_dma_yuv420p(in, s->src_dma.ptr, in_w, in_h);

        /* Mid-pipeline layout change (NV12 <-> YUV420P) would require
         * sws-based conversion either before or after V2D.  We do not
         * support it via V2D — fall back to libswscale. */
        if (in->format != s->out_format)
            goto sw_fallback;

        /* V2D scale: src_dma → dst_dma in whichever native layout */
        if (s->out_format == AV_PIX_FMT_NV12)
            ret = v2d_scale_nv12(avctx,
                                 s->src_dma.fd, in_w,    in_h,
                                 s->dst_dma.fd, s->out_w, s->out_h);
        else
            ret = v2d_scale_yuv420p(avctx,
                                    s->src_dma.fd, in_w,    in_h,
                                    s->dst_dma.fd, s->out_w, s->out_h);
        if (ret < 0) {
            av_log(avctx, AV_LOG_WARNING,
                   "scale_spacemit: V2D failed — disabling HW for this session\n");
            s->v2d_ok = 0;
            goto sw_fallback;
        }

        /* Allocate output AVFrame and copy from DMA buffer */
        out = ff_get_video_buffer(outlink, s->out_w, s->out_h);
        if (!out) { ret = AVERROR(ENOMEM); goto fail; }

        ret = av_frame_copy_props(out, in);
        if (ret < 0) goto fail;

        if (s->out_format == AV_PIX_FMT_NV12)
            dma_to_frame_nv12(s->dst_dma.ptr, out, s->out_w, s->out_h);
        else
            dma_to_frame_yuv420p(s->dst_dma.ptr, out, s->out_w, s->out_h);

        out->width  = s->out_w;
        out->height = s->out_h;
        out->format = s->out_format;

        /* Strip size-dependent side data when resolution changed */
        if (in->width != s->out_w || in->height != s->out_h) {
            av_frame_side_data_remove_by_props(&out->side_data,
                                               &out->nb_side_data,
                                               AV_SIDE_DATA_PROP_SIZE_DEPENDENT);
        }

        av_frame_free(&in);
        return ff_filter_frame(outlink, out);
    }

sw_fallback:
    /* ── libswscale CPU fallback ── */
    out = ff_get_video_buffer(outlink, s->out_w, s->out_h);
    if (!out) { ret = AVERROR(ENOMEM); goto fail; }

    ret = av_frame_copy_props(out, in);
    if (ret < 0) goto fail;

    out->width  = s->out_w;
    out->height = s->out_h;
    out->format = s->out_format;

    s->sws = sws_getCachedContext(s->sws,
        in->width, in->height, in->format,
        s->out_w,  s->out_h,   s->out_format,
        SWS_BILINEAR, NULL, NULL, NULL);
    if (!s->sws) { ret = AVERROR(ENOMEM); goto fail; }

    sws_scale(s->sws,
              (const uint8_t * const *)in->data, in->linesize,
              0, in->height,
              out->data, out->linesize);

    if (in->width != s->out_w || in->height != s->out_h) {
        av_frame_side_data_remove_by_props(&out->side_data,
                                           &out->nb_side_data,
                                           AV_SIDE_DATA_PROP_SIZE_DEPENDENT);
    }

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);

fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

/* ── AVFilter descriptor ───────────────────────────────────────────────── */

#define OFFSET(x) offsetof(ScaleSpacemitContext, x)
#define FLAGS (AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM)

static const AVOption scale_spacemit_options[] = {
    { "w",      "output width expression",
      OFFSET(w_expr), AV_OPT_TYPE_STRING, { .str = "iw" }, .flags = FLAGS },
    { "width",  "output width expression",
      OFFSET(w_expr), AV_OPT_TYPE_STRING, { .str = NULL  }, .flags = FLAGS },
    { "h",      "output height expression",
      OFFSET(h_expr), AV_OPT_TYPE_STRING, { .str = "ih" }, .flags = FLAGS },
    { "height", "output height expression",
      OFFSET(h_expr), AV_OPT_TYPE_STRING, { .str = NULL  }, .flags = FLAGS },
    { "force_original_aspect_ratio",
      "decrease or increase w/h if necessary to keep the original AR",
      OFFSET(force_original_aspect_ratio), AV_OPT_TYPE_INT,
      { .i64 = 0 }, 0, 2, FLAGS, .unit = "force_oar" },
    { "disable",  NULL, 0, AV_OPT_TYPE_CONST,
      { .i64 = 0 }, 0, 0, FLAGS, .unit = "force_oar" },
    { "decrease", NULL, 0, AV_OPT_TYPE_CONST,
      { .i64 = 1 }, 0, 0, FLAGS, .unit = "force_oar" },
    { "increase", NULL, 0, AV_OPT_TYPE_CONST,
      { .i64 = 2 }, 0, 0, FLAGS, .unit = "force_oar" },
    { "force_divisible_by",
      "enforce that the output resolution is divisible by the given integer "
      "when force_original_aspect_ratio is used",
      OFFSET(force_divisible_by), AV_OPT_TYPE_INT,
      { .i64 = 1 }, 1, 256, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(scale_spacemit);

static const AVFilterPad scale_spacemit_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = scale_spacemit_filter_frame,
    },
};

static const AVFilterPad scale_spacemit_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = scale_spacemit_config_output,
    },
};

const FFFilter ff_vf_scale_spacemit = {
    .p.name        = "scale_spacemit",
    .p.description = NULL_IF_CONFIG_SMALL(
        "Scale video using SpacemiT K1 V2D hardware blitter"),
    .p.priv_class  = &scale_spacemit_class,
    .p.flags       = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
    .priv_size     = sizeof(ScaleSpacemitContext),
    .init          = scale_spacemit_init,
    .uninit        = scale_spacemit_uninit,
    FILTER_INPUTS(scale_spacemit_inputs),
    FILTER_OUTPUTS(scale_spacemit_outputs),
    FILTER_QUERY_FUNC2(scale_spacemit_query_formats),
};
