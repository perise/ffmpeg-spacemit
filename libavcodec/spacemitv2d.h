/*
 * spacemitv2d.h — SpacemiT K1 V2D hardware blitter integration for FFmpeg
 *
 * Provides zero-copy frame transfer between the hardware decoder's V4L2
 * CAPTURE buffers and the hardware encoder's V4L2 OUTPUT buffers by:
 *
 *  1. Exporting V4L2 mmap buffers as DMA-buf fds (VIDIOC_EXPBUF)
 *  2. Using the V2D 2D blitter to copy/scale/convert entirely in hardware
 *     without touching the CPU data path
 *
 * The V2D device (/dev/v2d_dev) on K1 handles:
 *   - Format-aware NV12→NV12 bitblit (single fd contiguous NV12)
 *   - Y8 raw byte copy for 2-plane V4L2 layouts (separate Y + UV fds)
 *   - Optional CSC (BT.601/BT.709 wide/narrow conversions)
 *   - Scaling (when src_rect ≠ dst_rect)
 *   - Rotation (0°/90°/180°/270°/mirror/flip)
 *
 * This file is part of FFmpeg / ffmpeg-spacemit.
 */

#ifndef AVCODEC_SPACEMITV2D_H
#define AVCODEC_SPACEMITV2D_H

#include <stdint.h>
#include "libavutil/log.h"

/* DMA-buf fd set attached to an AVFrame produced by the zero-copy decoder.
 * Stored in frame->opaque so the encoder can retrieve it without an
 * intermediary copy. */
typedef struct SpacemitDMAFrameInfo {
    /* DMA-buf fds exported from the decoder's V4L2 CAPTURE buffer.
     * dma_fds[0] = Y plane fd (always valid)
     * dma_fds[1] = UV plane fd (valid only when n_planes == 2)
     *              For 1-plane (contiguous NV12) UV starts at offset y_size. */
    int dma_fds[3];
    int n_planes;           /* 1 (contiguous NV12) or 2 (separate Y+UV) */
    int width, height;
    int y_stride, uv_stride;
    int y_size;             /* width * height bytes */
    /* back-pointer to the AVBuffer that manages V4L2 re-queue lifecycle
     * (held by frame->buf[0]); used for assertion only. */
} SpacemitDMAFrameInfo;

/* V2D context — one per encoder instance. */
typedef struct SpacemitV2DCtx {
    void  *av_log_ctx;       /* for av_log() */
    int    available;        /* 1 if V2D device was opened successfully */
    /* internal — V2D device fd is managed inside spacemitv2d.c */
} SpacemitV2DCtx;

/* Open /dev/v2d_dev.  Returns 0 on success. */
int  spacemit_v2d_init(SpacemitV2DCtx *ctx, void *av_log_ctx);

/* Close the V2D device. */
void spacemit_v2d_close(SpacemitV2DCtx *ctx);

/* Blit from a decoded frame (described by src_info) into a V4L2 encoder
 * OUTPUT buffer (described by dst_* parameters).
 *
 * Handles both 1-plane (contiguous NV12) and 2-plane (separate Y+UV) source
 * layouts.  When src and dst dimensions differ, V2D performs scaling.
 *
 *  src_info   - DMA-buf info from the decoder frame
 *  dst_fds[]  - DMA-buf fds for encoder OUTPUT planes
 *  dst_nplanes - 1 (contiguous) or 2 (separate)
 *  dst_w/h    - encoder resolution
 *  dst_stride - encoder Y stride (UV stride = dst_stride)
 *
 * Returns 0 on success, negative on error (caller should fall back to CPU).
 */
int spacemit_v2d_blit(SpacemitV2DCtx       *ctx,
                      const SpacemitDMAFrameInfo *src_info,
                      const int            *dst_fds,
                      int                   dst_nplanes,
                      int                   dst_w,
                      int                   dst_h,
                      int                   dst_stride);

#endif /* AVCODEC_SPACEMITV2D_H */
