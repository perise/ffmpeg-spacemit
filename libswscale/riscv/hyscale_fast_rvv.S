/*
 * RISC-V RVV 1.0 implementation of fast bilinear horizontal scaler
 * for libswscale (hyscale_fast path, SWS_FAST_BILINEAR)
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

#include "config.h"

#if HAVE_RVV

.attribute arch, "rv64gcv_zba_zbb"

/*
 * void ff_hyscale_fast_rvv(SwsContext *c,
 *                           int16_t *dst, int dstW,
 *                           const uint8_t *src, int srcW, int xInc)
 *
 * Fast bilinear horizontal scaler for 8-bit luma input.
 * Used when SWS_FAST_BILINEAR flag is set.
 *
 * For each output pixel i (with xpos = i * xInc):
 *   xx     = xpos >> 16          (integer source index)
 *   xalpha = (xpos & 0xFFFF) >> 9  (blend factor, 7-bit, 0..127)
 *   dst[i] = (src[xx] << 7) + (src[xx+1] - src[xx]) * xalpha
 *          = src[xx] * (128 - xalpha) + src[xx+1] * xalpha
 *           (using 7-bit coefficients → result in 15-bit range)
 *
 * RVV Strategy (VLEN=256, e32m1 = 8 elements per register):
 *   1. Compute xpos[i] = i * xInc using vid.v + vmul.vx
 *   2. Extract xx[i] = xpos >> 16  (vsrl.vi with 16)
 *   3. Extract xalpha[i] = (xpos & 0xFFFF) >> 9  (vand + vsrl)
 *   4. Gather src[xx[i]] using e8mf4 + vluxei32.v (indices in e32m1)
 *   5. Gather src[xx[i]+1] similarly (after vadd.vi 1 to index)
 *   6. Compute blend in e32m1: dst = src0*(128-xalpha) + src1*xalpha
 *   7. Narrow to int16 and store
 *
 * LMUL constraints for indexed gather of uint8 with int32 indices, VL=8:
 *   - SEW=8, LMUL=mf4: max VL = VLEN/(8*4) = 256/32 = 8  ✓
 *   - EEW_index=32, LMUL_index = mf4 * (32/8) = m1       ✓
 *   So: vsetvli e8,mf4, vluxei32.v vd,(src),v_xx{e32m1}
 *
 * ABI (rv64):
 *   a0 = SwsContext *c (ignored)
 *   a1 = int16_t *dst
 *   a2 = int dstW
 *   a3 = const uint8_t *src
 *   a4 = int srcW (ignored)
 *   a5 = int xInc
 */

.globl ff_hyscale_fast_rvv
.type  ff_hyscale_fast_rvv, @function

ff_hyscale_fast_rvv:
    .cfi_startproc

    /* Quick exit if dstW <= 0 */
    blez a2, .Lhyf_done

    /* Save callee-saved registers we'll use */
    addi    sp, sp, -48
    sd      s0,  0(sp)
    sd      s1,  8(sp)
    sd      s2, 16(sp)
    sd      s3, 24(sp)
    sd      s4, 32(sp)
    sd      s5, 40(sp)
    .cfi_def_cfa_offset 48
    .cfi_offset s0,  0
    .cfi_offset s1,  8
    .cfi_offset s2, 16
    .cfi_offset s3, 24
    .cfi_offset s4, 32
    .cfi_offset s5, 40

    mv      s0, a1          /* dst */
    mv      s1, a2          /* dstW */
    mv      s2, a3          /* src */
    mv      s3, a5          /* xInc */
    li      s4, 0x0000FFFF  /* mask for fractional part */
    xor     s5, s5, s5      /* i = 0 (current output pixel index) */

.Lhyf_batch:
    sub     t0, s1, s5              /* remaining = dstW - i */
    vsetvli t1, t0, e32, m1, ta, ma /* t1 = VL ≤ 8 for e32m1, VLEN=256 */

    /*
     * Compute xpos[j] = (i + j) * xInc for j in [0, VL)
     *
     * vid.v gives [0, 1, 2, ..., VL-1]
     * Then add current base index i → [i, i+1, ..., i+VL-1]
     * Then multiply by xInc
     */
    vid.v   v0                      /* v0 = [0, 1, 2, ..., VL-1] (e32m1) */
    vadd.vx v0, v0, s5              /* v0 = [i, i+1, ..., i+VL-1] */
    vmul.vx v1, v0, s3              /* v1 = xpos[j] = (i+j) * xInc  (e32m1) */

    /*
     * Extract integer source position: xx[j] = xpos[j] >> 16
     * and blend factor: xalpha[j] = (xpos[j] & 0xFFFF) >> 9
     */
    vsrl.vi v2, v1, 16              /* v2 = xx[j]     (e32m1, unsigned) */
    vand.vx v3, v1, s4              /* v3 = xpos & 0xFFFF */
    vsrl.vi v3, v3, 9               /* v3 = xalpha[j] (0..127) */

    /*
     * Gather src0[j] = src[xx[j]] using indexed load.
     *
     * Switch to e8,mf4 for data (VL=8, LMUL=mf4).
     * Index vector v2 remains valid as e32m1 (LMUL_index = mf4 * 4 = m1).
     * vluxei32.v: loads 8 uint8 values at addresses (src + v2[j]).
     */
    vsetvli zero, t1, e8, mf4, ta, ma  /* VL=t1, e8, mf4 */
    vluxei32.v v4, (s2), v2         /* v4 = src[xx[j]]  (e8mf4) */

    /* Compute src1 index: xx[j] + 1 */
    vadd.vi v5, v2, 1               /* v5 = xx[j] + 1 (still e32m1 in register, but
                                       we're now in e8mf4 mode — v5 treated as mf4 here.
                                       But v2 was e32m1; reuse after mode switch is OK
                                       as long as we don't confuse widths.
                                       Actually we need to do vadd in e32 mode. */
    /* Re-set to e32m1 to compute xx+1, then back to e8mf4 for gather */
    vsetvli zero, t1, e32, m1, ta, ma  /* back to e32m1 */
    vadd.vi v5, v2, 1               /* v5 = xx[j] + 1  (e32m1) */
    vsetvli zero, t1, e8, mf4, ta, ma  /* back to e8mf4 for second gather */
    vluxei32.v v6, (s2), v5         /* v6 = src[xx[j]+1]  (e8mf4) */

    /*
     * Zero-extend gathered uint8 to int32 for arithmetic.
     * vzext.vf4: widens each e8mf4 element to e32m1 (4x width increase).
     *
     * RVV 1.0 constraint: for widening ops, destination and source register
     * groups MUST NOT overlap (spec §5.2).  Use v8/v9 as destinations so they
     * are distinct from the source registers v4/v6.
     */
    vsetvli zero, t1, e32, m1, ta, ma  /* back to e32m1 — destination SEW */
    vzext.vf4 v8, v4                /* v8 = (uint32)src[xx[j]]   (e32m1) */
    vzext.vf4 v9, v6                /* v9 = (uint32)src[xx[j]+1] (e32m1) */

    /*
     * Bilinear blend:
     *   diff[j] = src1[j] - src0[j]
     *   dst32[j] = (src0[j] << 7) + diff[j] * xalpha[j]
     *
     * All in e32m1.
     */
    vsub.vv v7, v9, v8              /* v7 = src1 - src0           (e32m1) */
    vmul.vv v7, v7, v3              /* v7 = diff * xalpha         (e32m1) */
    vsll.vi v8, v8, 7               /* v8 = src0 << 7             (e32m1) */
    vadd.vv v7, v8, v7              /* v7 = (src0<<7) + diff*alpha (e32m1) */

    /*
     * Narrow int32 → int16 and store.
     * vnsrl.wi: arithmetic right-shift and narrow e32→e16.
     * shift=0: just narrow (truncate to lower 16 bits, unsigned).
     * The result is in range [0, 32767] for valid 8-bit input, so no overflow.
     *
     * After vnsrl, we get e16mf2 (half the register, VL=t1 e16 elements).
     * Store to dst[i..i+VL-1].
     */
    vsetvli zero, t1, e16, mf2, ta, ma /* switch to e16mf2 for narrowing */
    vnsrl.wi v8, v7, 0              /* v8 = (int16)v7[j] = dst values  (e16mf2) */

    /* Store to dst: 2 bytes per element, VL elements */
    slli    t2, s5, 1               /* t2 = i * 2 (byte offset) */
    add     t2, s0, t2              /* t2 = &dst[i] */
    vse16.v v8, (t2)                /* store VL int16 values */

    /* Advance i by VL */
    add     s5, s5, t1
    blt     s5, s1, .Lhyf_batch

.Lhyf_done:
    ld      s0,  0(sp)
    ld      s1,  8(sp)
    ld      s2, 16(sp)
    ld      s3, 24(sp)
    ld      s4, 32(sp)
    ld      s5, 40(sp)
    addi    sp, sp, 48
    ret

    .cfi_endproc
.size ff_hyscale_fast_rvv, .-ff_hyscale_fast_rvv

/* =========================================================================
 * ff_hcscale_fast_rvv — fast bilinear scaler for interleaved chroma (NV12 UV)
 *
 * void ff_hcscale_fast_rvv(SwsContext *c,
 *                            int16_t *dst1, int16_t *dst2,
 *                            int dstW,
 *                            const uint8_t *src1, const uint8_t *src2,
 *                            int srcW, int xInc)
 *
 * Scales two chroma planes simultaneously using the same xpos sequence.
 * For NV12 input, src1 and src2 point to U and V planes respectively.
 *
 * ABI (rv64):
 *   a0 = SwsContext *c (ignored)
 *   a1 = int16_t *dst1 (U output)
 *   a2 = int16_t *dst2 (V output)
 *   a3 = int dstW
 *   a4 = const uint8_t *src1 (U input)
 *   a5 = const uint8_t *src2 (V input)
 *   a6 = int srcW (ignored)
 *   a7 = int xInc
 * ========================================================================= */

.globl ff_hcscale_fast_rvv
.type  ff_hcscale_fast_rvv, @function

ff_hcscale_fast_rvv:
    .cfi_startproc

    blez    a3, .Lhcf_done

    addi    sp, sp, -64
    sd      s0,  0(sp)
    sd      s1,  8(sp)
    sd      s2, 16(sp)
    sd      s3, 24(sp)
    sd      s4, 32(sp)
    sd      s5, 40(sp)
    sd      s6, 48(sp)
    sd      s7, 56(sp)
    .cfi_def_cfa_offset 64

    mv      s0, a1          /* dst1 (U) */
    mv      s1, a2          /* dst2 (V) */
    mv      s2, a3          /* dstW */
    mv      s3, a4          /* src1 (U) */
    mv      s4, a5          /* src2 (V) */
    mv      s5, a7          /* xInc */
    li      s6, 0x0000FFFF  /* 0xFFFF mask */
    xor     s7, s7, s7      /* i = 0 */

.Lhcf_batch:
    sub     t0, s2, s7
    vsetvli t1, t0, e32, m1, ta, ma

    /* xpos[j] = (i+j) * xInc */
    vid.v   v0
    vadd.vx v0, v0, s7
    vmul.vx v1, v0, s5              /* xpos */
    vsrl.vi v2, v1, 16              /* xx */
    vand.vx v3, v1, s6
    vsrl.vi v3, v3, 9               /* xalpha */

    /* Compute xx+1 in e32m1 */
    vadd.vi v5, v2, 1               /* xx+1 */

    /* === U plane === */
    vsetvli zero, t1, e8, mf4, ta, ma
    vluxei32.v v4, (s3), v2         /* src1[xx]   → U (e8mf4) */
    vluxei32.v v6, (s3), v5         /* src1[xx+1] → U+1 (e8mf4) */

    /* Widen to e32m1; use v10/v11 to avoid same-register overlap (RVV §5.2) */
    vsetvli zero, t1, e32, m1, ta, ma
    vzext.vf4 v10, v4               /* v10 = (u32)U[xx]   */
    vzext.vf4 v11, v6               /* v11 = (u32)U[xx+1] */
    vsub.vv v7, v11, v10
    vmul.vv v7, v7, v3
    vsll.vi v10, v10, 7
    vadd.vv v8, v10, v7             /* v8 = U blend result */

    /* === V plane === */
    vsetvli zero, t1, e8, mf4, ta, ma
    vluxei32.v v4, (s4), v2         /* src2[xx]   → V (e8mf4) */
    vluxei32.v v6, (s4), v5         /* src2[xx+1] → V+1 (e8mf4) */

    /* Widen to e32m1; reuse v10/v11 (U blend result already in v8) */
    vsetvli zero, t1, e32, m1, ta, ma
    vzext.vf4 v10, v4               /* v10 = (u32)V[xx]   */
    vzext.vf4 v11, v6               /* v11 = (u32)V[xx+1] */
    vsub.vv v7, v11, v10
    vmul.vv v7, v7, v3
    vsll.vi v10, v10, 7
    vadd.vv v9, v10, v7             /* v9 = V blend result */

    /* Narrow and store both planes */
    vsetvli zero, t1, e16, mf2, ta, ma
    vnsrl.wi v10, v8, 0             /* U int16 */
    vnsrl.wi v11, v9, 0             /* V int16 */

    slli    t2, s7, 1
    add     t3, s0, t2              /* &dst1[i] */
    add     t4, s1, t2              /* &dst2[i] */
    vse16.v v10, (t3)
    vse16.v v11, (t4)

    add     s7, s7, t1
    blt     s7, s2, .Lhcf_batch

.Lhcf_done:
    ld      s0,  0(sp)
    ld      s1,  8(sp)
    ld      s2, 16(sp)
    ld      s3, 24(sp)
    ld      s4, 32(sp)
    ld      s5, 40(sp)
    ld      s6, 48(sp)
    ld      s7, 56(sp)
    addi    sp, sp, 64
    ret

    .cfi_endproc
.size ff_hcscale_fast_rvv, .-ff_hcscale_fast_rvv

#endif /* HAVE_RVV */
