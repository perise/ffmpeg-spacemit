/*
 * RVV horizontal 8-to-15 bpc scaler for SpacemiT K1 (VLEN=256)
 *
 * Copyright (C) 2024 perise
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This file is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/*
 * Implements ff_hscale8to15_rvv - replaces both hyScale and hcScale.
 *
 * Algorithm (matches hScale8To15_c):
 *   for i in 0..dstW-1:
 *     val = sum(src[filterPos[i]+j] * filter[i*filterSize+j], j=0..filterSize-1)
 *     dst[i] = FFMIN(val >> 7, (1<<15)-1)
 *
 * RVV strategy (e32m4, 32 output pixels per outer iteration with VLEN=256):
 *   - Outer loop: load 32 filterPos values (e32m4)
 *   - Inner j-loop:
 *       positions = filterPos + j          (vadd.vx e32m4)
 *       gather src bytes                   (vluxei32.v e8m1, indices=v8-v11 e32m4)
 *       widen to e32m4                     (vzext.vf4 v15, v12 -- v15 != v12 no SIGILL)
 *       strided-load filter coefficients   (vlse16.v e16m2, stride=filterSize*2)
 *       widen to e32m4                     (vsext.vf2 v19, v13 -- v19 != v13 no SIGILL)
 *       accumulate                         (vmacc.vv v0, v15, v19 -- e32m4 no widening)
 *   - Narrow e32m4 -> e16m2                (vnsra.wi shift-7, arithmetic, exact C match)
 *   - Store e16m2 to dst
 *
 * Register map (VLEN=256, LMUL=4, 32 elements per group):
 *   v0-v3   accumulator (e32m4)
 *   v4-v7   filterPos base values (e32m4, unchanged through inner loop)
 *   v8-v11  positions = filterPos+j (e32m4); also vluxei32.v index group
 *   v12     gathered source bytes (e8m1); EMUL_index=32/8*1=4 -> indices v8-v11
 *   v13-v14 filter coefficients int16 (e16m2)
 *   v15-v18 zero-widened source uint32 (e32m4)
 *   v19-v22 sign-widened filter int32 (e32m4)
 *   v23-v24 narrowed output int16 (e16m2)
 *
 * Prototype:
 *   void ff_hscale8to15_rvv(SwsContext *c, int16_t *dst, int dstW,
 *                            const uint8_t *src, const int16_t *filter,
 *                            const int32_t *filterPos, int filterSize);
 * Arguments:
 *   a0 = c         (SwsContext*, unused)
 *   a1 = dst       (int16_t*)
 *   a2 = dstW      (int, used as remaining)
 *   a3 = src       (const uint8_t*)
 *   a4 = filter    (const int16_t*, advances by vl*filterSize*2 per outer)
 *   a5 = filterPos (const int32_t*, advances by vl*4 per outer)
 *   a6 = filterSize (int)
 */

#include "libavutil/riscv/asm.S"

func ff_hscale8to15_rvv, zve32x, zba
        lpad    0
        /* t0 = filterSize * 2  (byte stride between rows in filter table) */
        slli    t0, a6, 1

.L_outer:
        /* VL = min(remaining, 32) for e32m4 with VLEN=256 */
        vsetvli a7, a2, e32, m4, ta, ma
        beqz    a7, .L_done

        /* Load filterPos[i .. i+vl-1] into v4-v7 (constant through j loop) */
        vle32.v v4, (a5)

        /* Clear accumulator */
        vmv.v.i v0, 0

        /* t2 = filter_j_ptr: filter[outer_base * filterSize + j], j starts at 0 */
        mv      t2, a4
        /* t1 = j = 0 */
        li      t1, 0

.L_inner:
        /* Step 1: positions = filterPos[i] + j */
        vadd.vx v8, v4, t1          /* v8-v11 = filterPos[i]+j  (e32m4) */

        /* Step 2: gather uint8 source bytes at positions */
        /* SEW=8, EMUL_data=1 -> vd=v12; EMUL_index=32/8*1=4 -> vs2=v8-v11 */
        vsetvli zero, a7, e8, m1, ta, ma
        vluxei32.v v12, (a3), v8    /* v12 = gathered bytes (v12 not in v8-v11) */

        /* Step 3: zero-extend gathered bytes uint8 -> uint32 */
        vsetvli zero, a7, e32, m4, ta, ma
        vzext.vf4 v15, v12          /* v15-v18 (v15 != v12 -- no SIGILL) */

        /* Step 4: strided load filter[i*filterSize+j] for i=0..vl-1 */
        /* stride = filterSize*2 bytes (consecutive i values, fixed j column) */
        vsetvli zero, a7, e16, m2, ta, ma
        vlse16.v v13, (t2), t0      /* v13-v14 = int16 filter coefficients */

        /* Step 5: sign-extend filter int16 -> int32 and accumulate */
        vsetvli zero, a7, e32, m4, ta, ma
        vsext.vf2 v19, v13          /* v19-v22 (v19 != v13 -- no SIGILL) */
        vmacc.vv  v0, v15, v19      /* v0-v3 += v15-v18 * v19-v22  (no widening) */

        /* Advance j */
        addi    t1, t1, 1
        addi    t2, t2, 2           /* filter_j_ptr += sizeof(int16_t) */
        blt     t1, a6, .L_inner

        /* Narrow: e32m4 accumulator -> e16m2, arithmetic right shift 7 */
        /* In e16m2 mode, vnsra.wi: vd=e16 (v23-v24), vs2=e32 (v0-v3) */
        vsetvli zero, a7, e16, m2, ta, ma
        vnsra.wi  v23, v0, 7        /* v23-v24 = int16 output (v23 != v0 -- no SIGILL) */
        vse16.v   v23, (a1)

        /* Advance dst by vl * sizeof(int16_t) */
        slli    t3, a7, 1
        add     a1, a1, t3

        /* Advance filterPos by vl * sizeof(int32_t) */
        slli    t3, a7, 2
        add     a5, a5, t3

        /* Advance filter base by vl * filterSize * sizeof(int16_t) = vl * t0 */
        mul     t3, a7, t0
        add     a4, a4, t3

        /* remaining -= vl */
        sub     a2, a2, a7
        bnez    a2, .L_outer

.L_done:
        ret
endfunc
