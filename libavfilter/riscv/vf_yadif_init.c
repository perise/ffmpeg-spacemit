/*
 * RVV yadif deinterlace init for SpacemiT K1
 * Copyright (C) 2024 perise
 *
 * This file is part of FFmpeg.
 * LGPL 2.1 or later.
 */

#include "config.h"
#include "libavutil/attributes.h"
#include "libavutil/riscv/cpu.h"
#include "libavfilter/yadif.h"

void ff_yadif_filter_line_8_rvv(void *dst, void *prev, void *cur, void *next,
                                 int w, int prefs, int mrefs,
                                 int parity, int mode);

/* Wrapper: filter_line_c calls us with (dst, prev, cur, next, w, prefs, mrefs, parity, mode).
 * The RVV kernel matches exactly -- just forward the call. */
static void yadif_filter_line_8_rvv_wrapper(void *dst, void *prev, void *cur,
                                             void *next, int w, int prefs,
                                             int mrefs, int parity, int mode)
{
    ff_yadif_filter_line_8_rvv(dst, prev, cur, next, w, prefs, mrefs, parity, mode);
}

av_cold void ff_yadif_init_riscv(YADIFContext *s)
{
#if HAVE_RVV
    int flags = av_get_cpu_flags();
    if ((flags & AV_CPU_FLAG_RVV_I32) && (flags & AV_CPU_FLAG_RVB)) {
        /* 8-bit planar formats only; 16-bit falls back to C */
        if (s->csp && s->csp->comp[0].depth == 8)
            s->filter_line = yadif_filter_line_8_rvv_wrapper;
    }
#endif
}
