/*
 * RISC-V RVV 1.0 implementation of overlay alpha-blend row functions
 * for libavfilter/vf_overlay (straight alpha, 8-bit YUV)
 *
 * Copyright (c) 2026 SpacemiT K1 optimization
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

#include "libavutil/riscv/asm.S"

/*
 * All three blend_row functions implement the same per-pixel formula:
 *
 *   dst[x] = FAST_DIV255(dst[x] * (255 - alpha) + src[x] * alpha)
 *
 * where FAST_DIV255(x) = ((x + 128 + ((x + 128) >> 8)) >> 8)
 *                       ≈ x / 255  (error ≤ 1 ULP)
 *
 * This matches the x86 SSE4 blend formula:
 *   pmulhuw(s*alpha + d*(255-alpha) + 128,  pw_257)
 *
 * RVV strategy (e8m1 → 32 pixels/VL at VLEN=256):
 *   1. Load dst, src  (e8m1)
 *   2. Load / compute alpha, alpha_inv = 255 - alpha  (e8m1)
 *   3. vwmulu.vv  v_prod, dst, alpha_inv      → e16m2
 *   4. vwmaccu.vv v_prod, src, alpha          → += src×alpha
 *   5. FAST_DIV255 in e16m2:
 *        vadd.vx  v_prod, v_prod, t_128
 *        vsrl.vi  v_tmp,  v_prod, 8
 *        vadd.vv  v_prod, v_prod, v_tmp
 *        vsrl.vi  v_prod, v_prod, 8         → [0..255]
 *   6. vnclipu.wi v_dst, v_prod, 0          → e16m2 → e8m1 (saturating)
 *   7. vse8.v     v_dst, (d)
 *
 * Function signature (identical for all three, matches x86 blend_row):
 *   int func(uint8_t *d, uint8_t *da, uint8_t *s, uint8_t *a,
 *             int w, ptrdiff_t alinesize)
 *   Returns w (all pixels processed by the RVV loop).
 *
 * ABI:
 *   a0 = d  (dst pixels, in/out)
 *   a1 = da (dst alpha — unused for straight blend)
 *   a2 = s  (src overlay pixels)
 *   a3 = a  (alpha plane pointer)
 *   a4 = w  (pixel count)
 *   a5 = alinesize  (alpha stride — used only by blend_row_20)
 *
 * Temporaries used (all caller-saved, no save/restore needed):
 *   t0 = VL from vsetvli
 *   t3 = constant 255  (loaded once before loop)
 *   t4 = constant 128  (loaded once before loop)
 *   t6 = saved w for return value
 */

/* =========================================================================
 * ff_overlay_blend_row_44_rvv  —  non-subsampled alpha (stride 1 per pixel)
 *
 * Used for:
 *   - All planes of YUV444 / GBRP
 *   - Luma (Y) plane of YUV420 / YUV422
 *
 * Speedup vs scalar: ~8× on K1 (32 px/iter vs ~4 px scalar throughput).
 * ========================================================================= */

func ff_overlay_blend_row_44_rvv, zve32x, zba
        lpad    0

        mv      t6, a4              /* t6 = w (return value) */
        li      t3, 255             /* t3 = 255 constant */
        li      t4, 128             /* t4 = 128 bias for FAST_DIV255 */

.L44_loop:
        vsetvli t0, a4, e8, m1, ta, ma   /* VL = min(w, 32) */

        vle8.v  v0, (a0)                  /* dst pixels */
        vle8.v  v1, (a2)                  /* src pixels */
        vle8.v  v2, (a3)                  /* alpha[x] */

        vrsub.vx v3, v2, t3               /* v3 = 255 - alpha */

        /* Widening multiply-accumulate → e16m2  */
        vwmulu.vv  v4, v0, v3             /* v4 = dst * (255-alpha) */
        vwmaccu.vv v4, v1, v2             /* v4 += src * alpha */

        /* FAST_DIV255 in e16m2 */
        vsetvli zero, t0, e16, m2, ta, ma
        vadd.vx v4, v4, t4               /* + 128 */
        vsrl.vi v6, v4, 8                /* >> 8 */
        vadd.vv v4, v4, v6               /* + (>>8) */
        vsrl.vi v4, v4, 8               /* >> 8 → [0..255] */

        /* Narrow back to e8m1 and store */
        vsetvli zero, t0, e8, m1, ta, ma
        vnclipu.wi v0, v4, 0
        vse8.v  v0, (a0)

        add     a0, a0, t0
        add     a2, a2, t0
        add     a3, a3, t0
        sub     a4, a4, t0
        bnez    a4, .L44_loop

        mv      a0, t6
        ret
endfunc

/* =========================================================================
 * ff_overlay_blend_row_22_rvv  —  horizontally subsampled alpha (hsub=1)
 *
 * Used for:
 *   - Chroma (U, V) planes of YUV422 overlay
 *
 * Input alpha: interleaved pairs [α_even, α_odd] per chroma pixel.
 * Uses vlseg2e8.v to deinterleave in one instruction.
 * vaaddu.vv computes rounded average: (α_even + α_odd + 1) >> 1.
 * ========================================================================= */

func ff_overlay_blend_row_22_rvv, zve32x, zba
        lpad    0

        mv      t6, a4
        li      t3, 255
        li      t4, 128

.L22_loop:
        vsetvli t0, a4, e8, m1, ta, ma

        vle8.v  v0, (a0)                  /* dst */
        vle8.v  v1, (a2)                  /* src */

        /*
         * vlseg2e8.v with NFIELDS=2, e8m1:
         *   Loads 2×VL bytes from a3 interleaved:
         *   v2[i] = a[2i]   (even bytes = first of each pair)
         *   v3[i] = a[2i+1] (odd bytes  = second of each pair)
         */
        vlseg2e8.v v2, (a3)

        /* alpha = round_avg(α_even, α_odd) */
        vaaddu.vv  v2, v2, v3             /* v2 = (α_even + α_odd + 1) >> 1 */

        vrsub.vx   v3, v2, t3             /* 255 - alpha */
        vwmulu.vv  v4, v0, v3
        vwmaccu.vv v4, v1, v2

        vsetvli zero, t0, e16, m2, ta, ma
        vadd.vx v4, v4, t4
        vsrl.vi v6, v4, 8
        vadd.vv v4, v4, v6
        vsrl.vi v4, v4, 8

        vsetvli zero, t0, e8, m1, ta, ma
        vnclipu.wi v0, v4, 0
        vse8.v  v0, (a0)

        add        a0, a0, t0
        add        a2, a2, t0
        sh1add     a3, t0, a3            /* a += 2*VL (2 alpha bytes per chroma pixel) */
        sub        a4, a4, t0
        bnez       a4, .L22_loop

        mv      a0, t6
        ret
endfunc

/* =========================================================================
 * ff_overlay_blend_row_20_rvv  —  2×2 subsampled alpha (hsub=1, vsub=1)
 *
 * Used for:
 *   - Chroma (U, V) planes of YUV420 overlay
 *
 * Alpha is averaged over a 2×2 luma-alpha block:
 *   avg( a[2x],              a[2x+1],
 *        a[2x + alinesize],  a[2x+1 + alinesize] )
 *
 * Computed as avg(row0_avg, row1_avg) using two vlseg2e8.v loads.
 * Rounding matches x86 SSE4 overlay_row_20 (two pavgw steps).
 *
 * t1 = row1_ptr = a3 + alinesize  (recomputed each iteration since a3 moves)
 * ========================================================================= */

func ff_overlay_blend_row_20_rvv, zve32x, zba
        lpad    0

        mv      t6, a4
        li      t3, 255
        li      t4, 128

.L20_loop:
        vsetvli t0, a4, e8, m1, ta, ma

        vle8.v  v0, (a0)                  /* dst */
        vle8.v  v1, (a2)                  /* src */

        /* Row 0 alpha pairs */
        vlseg2e8.v v2, (a3)               /* v2=row0_even, v3=row0_odd */

        /* Row 1 alpha pairs: a3 + alinesize */
        add        t1, a3, a5
        vlseg2e8.v v8, (t1)              /* v8=row1_even, v9=row1_odd */

        /* Average within each row, then average the two row averages */
        vaaddu.vv  v2, v2, v3            /* row0_avg = avg(row0_even, row0_odd) */
        vaaddu.vv  v8, v8, v9            /* row1_avg = avg(row1_even, row1_odd) */
        vaaddu.vv  v2, v2, v8            /* alpha_avg = avg(row0_avg, row1_avg) */

        vrsub.vx   v3, v2, t3
        vwmulu.vv  v4, v0, v3
        vwmaccu.vv v4, v1, v2

        vsetvli zero, t0, e16, m2, ta, ma
        vadd.vx v4, v4, t4
        vsrl.vi v6, v4, 8
        vadd.vv v4, v4, v6
        vsrl.vi v4, v4, 8

        vsetvli zero, t0, e8, m1, ta, ma
        vnclipu.wi v0, v4, 0
        vse8.v  v0, (a0)

        add        a0, a0, t0
        add        a2, a2, t0
        sh1add     a3, t0, a3            /* a += 2*VL */
        sub        a4, a4, t0
        bnez       a4, .L20_loop

        mv      a0, t6
        ret
endfunc
