/*
 * spacemitv2d.c — SpacemiT K1 V2D hardware blitter for FFmpeg transcode
 *
 * Uses the system-installed libv2d.so (package: k1x-v2d) via the
 * /usr/include/spacemit/v2d_api.h public header.  The library exports
 * functions with an "ASR_V2D_" prefix; v2d_compat.h maps them to the
 * "V2D_" names declared in v2d_api.h.
 *
 * To enable: ./configure --enable-spacemit_v2d
 * Requires:  /usr/include/spacemit/v2d_api.h  +  /usr/lib/libv2d.so
 *
 * This file is part of FFmpeg / ffmpeg-spacemit.
 */

#include <errno.h>
#include <fcntl.h>
#ifndef O_CLOEXEC
#  define O_CLOEXEC 02000000
#endif
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* v2d_compat.h must come BEFORE v2d_api.h: it defines macros that map
 * V2D_BeginJob -> ASR_V2D_BeginJob etc., so the declarations in v2d_api.h
 * are processed with the correct (ASR_-prefixed) symbol names. */
#include <spacemit/v2d_compat.h>
#include <spacemit/v2d_type.h>
#include <spacemit/v2d_api.h>

#include "spacemitv2d.h"

/* ── helpers ────────────────────────────────────────────────────────────── */

static void fill_y8_surface(V2D_SURFACE_S *surf,
                             int fd, int offset,
                             int w, int h, int stride)
{
    memset(surf, 0, sizeof(*surf));
    surf->fbc_enable = 0;
    surf->fd         = fd;
    surf->offset     = offset;
    surf->w          = (uint16_t)w;
    surf->h          = (uint16_t)h;
    surf->stride     = (uint16_t)stride;
    surf->format     = V2D_COLOR_FORMAT_Y8;
}

static void fill_area(V2D_AREA_S *a, int w, int h)
{
    a->x = 0; a->y = 0;
    a->w = (uint16_t)w;
    a->h = (uint16_t)h;
}

/* ── public API ─────────────────────────────────────────────────────────── */

int spacemit_v2d_init(SpacemitV2DCtx *ctx, void *av_log_ctx)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->av_log_ctx = av_log_ctx;

    /* Probe: open device to confirm it's accessible, then close.
     * libv2d.so manages its own device fd internally. */
    int fd = open("/dev/v2d_dev", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        av_log(av_log_ctx, AV_LOG_WARNING,
               "spacemit_v2d: /dev/v2d_dev unavailable (%s) — CPU fallback\n",
               strerror(errno));
        ctx->available = 0;
        return -1;
    }
    close(fd);

    ctx->available = 1;
    av_log(av_log_ctx, AV_LOG_INFO,
           "spacemit_v2d: V2D hardware blitter enabled\n");
    return 0;
}

void spacemit_v2d_close(SpacemitV2DCtx *ctx)
{
    ctx->available = 0;
}

/*
 * spacemit_v2d_blit — copy NV12 frame (described by src_info DMA fds) to the
 * encoder's OUTPUT buffer (described by dst_fds[]).
 *
 * NV12 is handled as two separate Y8 bitblit tasks (one for Y, one for UV)
 * so that per-plane DMA-buf fds are supported even when Y and UV are in
 * separate V4L2 planes.
 */
int spacemit_v2d_blit(SpacemitV2DCtx            *ctx,
                      const SpacemitDMAFrameInfo *src,
                      const int                  *dst_fds,
                      int                         dst_nplanes,
                      int                         dst_w,
                      int                         dst_h,
                      int                         dst_stride)
{
    if (!ctx->available)
        return -1;

    V2D_HANDLE job;
    V2D_SURFACE_S s_surf, d_surf;
    V2D_AREA_S    s_area, d_area;

    int dst_y_fd  = dst_fds[0];
    int dst_uv_fd = (dst_nplanes >= 2) ? dst_fds[1] : dst_fds[0];
    int dst_y_off  = 0;
    int dst_uv_off = (dst_nplanes >= 2) ? 0 : dst_w * dst_h;

    int src_y_fd   = src->dma_fds[0];
    int src_uv_fd  = (src->n_planes >= 2) ? src->dma_fds[1] : src->dma_fds[0];
    int src_y_off  = 0;
    int src_uv_off = (src->n_planes >= 2) ? 0 : src->y_size;

    int src_uv_h = src->height / 2;
    int dst_uv_h = dst_h       / 2;

    if (V2D_BeginJob(&job) != SUCCESS) {
        av_log(ctx->av_log_ctx, AV_LOG_WARNING,
               "spacemit_v2d: V2D_BeginJob failed\n");
        return -1;
    }

    /* Y plane */
    fill_y8_surface(&s_surf, src_y_fd, src_y_off,
                    src->width, src->height, src->y_stride);
    fill_y8_surface(&d_surf, dst_y_fd, dst_y_off,
                    dst_w, dst_h, dst_stride);
    fill_area(&s_area, src->width, src->height);
    fill_area(&d_area, dst_w, dst_h);
    if (V2D_AddBitblitTask(job, &d_surf, &d_area, &s_surf, &s_area,
                           V2D_CSC_MODE_RGB_2_RGB) != SUCCESS)
        goto fail;

    /* UV plane */
    fill_y8_surface(&s_surf, src_uv_fd, src_uv_off,
                    src->width, src_uv_h, src->uv_stride);
    fill_y8_surface(&d_surf, dst_uv_fd, dst_uv_off,
                    dst_w, dst_uv_h, dst_stride);
    fill_area(&s_area, src->width, src_uv_h);
    fill_area(&d_area, dst_w, dst_uv_h);
    if (V2D_AddBitblitTask(job, &d_surf, &d_area, &s_surf, &s_area,
                           V2D_CSC_MODE_RGB_2_RGB) != SUCCESS)
        goto fail;

    if (V2D_EndJob(job) != SUCCESS) {
        av_log(ctx->av_log_ctx, AV_LOG_WARNING,
               "spacemit_v2d: V2D_EndJob failed\n");
        return -1;
    }
    return 0;

fail:
    V2D_EndJob(job);
    av_log(ctx->av_log_ctx, AV_LOG_WARNING,
           "spacemit_v2d: V2D_AddBitblitTask failed\n");
    return -1;
}
