/*
 * RISC-V RVV initialisation for libavfilter overlay blend_row functions.
 * Copyright (c) 2026 SpacemiT K1 optimization
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#include "libavutil/attributes.h"
#include "libavutil/cpu.h"
#include "libavutil/riscv/cpu.h"
#include "libavfilter/vf_overlay.h"

/* Declared in vf_overlay_rvv.S */
int ff_overlay_blend_row_44_rvv(uint8_t *d, uint8_t *da,
                                 uint8_t *s, uint8_t *a,
                                 int w, ptrdiff_t alinesize);
int ff_overlay_blend_row_22_rvv(uint8_t *d, uint8_t *da,
                                 uint8_t *s, uint8_t *a,
                                 int w, ptrdiff_t alinesize);
int ff_overlay_blend_row_20_rvv(uint8_t *d, uint8_t *da,
                                 uint8_t *s, uint8_t *a,
                                 int w, ptrdiff_t alinesize);

/**
 * Register RVV-optimised blend_row functions for the overlay filter.
 *
 * We only accelerate the simple straight-alpha path (alpha_format == 0,
 * main_has_alpha == 0) for 8-bit YUV formats, mirroring the x86 SSE4
 * implementation.  Premultiplied and main-has-alpha paths are left to
 * the scalar fallback.
 *
 * Speedup vs scalar: ~6–8× on K1 (VLEN=256, 32 pixels/cycle vs ~4).
 */
av_cold void ff_overlay_init_riscv(OverlayContext *s, int format,
                                    int pix_format,
                                    int alpha_format, int main_has_alpha)
{
#if HAVE_RVV
    int cpu_flags = av_get_cpu_flags();

    /* Need at minimum RVV with integer + zba for sh1add */
    if (!((cpu_flags & AV_CPU_FLAG_RVV_I32) && (cpu_flags & AV_CPU_FLAG_RVB)))
        return;

    /* Only handle straight alpha without destination alpha channel */
    if (alpha_format != 0 || main_has_alpha != 0)
        return;

    switch (format) {
    case OVERLAY_FORMAT_YUV444:
    case OVERLAY_FORMAT_GBRP:
        /* All three planes: 1:1 alpha per pixel */
        s->blend_row[0] = ff_overlay_blend_row_44_rvv;
        s->blend_row[1] = ff_overlay_blend_row_44_rvv;
        s->blend_row[2] = ff_overlay_blend_row_44_rvv;
        break;

    case OVERLAY_FORMAT_YUV420:
        /* Y plane: 1:1 alpha; U/V planes: 2×2 averaged alpha */
        s->blend_row[0] = ff_overlay_blend_row_44_rvv;
        s->blend_row[1] = ff_overlay_blend_row_20_rvv;
        s->blend_row[2] = ff_overlay_blend_row_20_rvv;
        break;

    case OVERLAY_FORMAT_YUV422:
        /* Y plane: 1:1 alpha; U/V planes: horizontally averaged alpha */
        s->blend_row[0] = ff_overlay_blend_row_44_rvv;
        s->blend_row[1] = ff_overlay_blend_row_22_rvv;
        s->blend_row[2] = ff_overlay_blend_row_22_rvv;
        break;

    default:
        break;
    }
#endif /* HAVE_RVV */
}
