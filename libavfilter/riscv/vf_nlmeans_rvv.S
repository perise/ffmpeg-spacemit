/*
 * RVV 1.0 nlmeans kernels for SpacemiT K1 (VLEN=256, zve32f)
 * Copyright (C) 2024 perise
 *
 * Two kernels:
 *
 * 1. ff_compute_safe_ssd_integral_image_rvv:
 *    For each row/col: dst[x] = dst_top[x] - dst_top[x-1] + (s1[x]-s2[x])^2 + dst[x-1]
 *    Strategy: vectorize sq_diff (e8m1→e16m2→e32m4), scalar carry prefix sum.
 *
 *    Register map (col loop):
 *      s0: carry (prefix sum accumulator; callee-saved)
 *      s1: temp ptr in carry loop (callee-saved)
 *      t0: VL from vsetvli; used as temp in carry loop then re-queried
 *      t1: temp address / carry loop counter / u32 stride
 *      t2: s1 col ptr (source row 1)
 *      t3: s2 col ptr (source row 2)
 *      t4: dst col ptr
 *      t5: dst_top col ptr
 *      t6: remaining pixels
 *
 *    RVV register alignment rules enforced (LMUL=4 needs 4-aligned, LMUL=2 needs 2-aligned):
 *      e8m1:   v2, v4 (any reg fine for m1)
 *      e16m2:  v12-v13 (sq_diff s1), v14-v15 (sq_diff s2)  [12,14 are even ✓]
 *      e32m4:  v0-v3 (widened sq), v4-v7 (dst_top), v8-v11 (dst_diag) [0,4,8 are /4 ✓]
 *
 * 2. ff_compute_weights_line_rvv:
 *    patch_diff_sq = FFMIN(iie[x]-iid[x]-iib[x]+iia[x], max)
 *    weight = weight_lut[patch_diff_sq]
 *    total_weight[x] += weight; sum[x] += weight * src[x]
 *    Strategy: fully vectorized (e32m4 = 32 pixels/iter) using vluxei32.v for lut gather.
 */

#include "libavutil/riscv/asm.S"

/*
 * void ff_compute_safe_ssd_integral_image_rvv(
 *     uint32_t *dst, ptrdiff_t dst_linesize_32,
 *     const uint8_t *s1, ptrdiff_t linesize1,
 *     const uint8_t *s2, ptrdiff_t linesize2,
 *     int w, int h)
 * a0=dst, a1=dst_linesize_32, a2=s1, a3=linesize1, a4=s2, a5=linesize2, a6=w, a7=h
 *
 * Preconditions: w >= 16 (multiple of 16), h >= 1.
 * Caller must zero dst[-1] and dst_top[-1] (border elements) before calling.
 *
 * VL note: VLEN=256, e8m1 VLMAX = e32m4 VLMAX = 32, so re-querying vsetvli
 * with e32m4 after the carry loop yields the same VL as the initial e8m1 query.
 */
func ff_compute_safe_ssd_integral_image_rvv, zve32x, zba
        lpad    0
        addi    sp, sp, -16
        sd      s0,  0(sp)
        sd      s1,  8(sp)

.L_ssd_row:
        beqz    a7, .L_ssd_done

        /* Compute dst_top_col = dst_col - dst_linesize_32*4 */
        slli    t0, a1, 2           /* t0 = dst_linesize_32 * 4 bytes */
        mv      t2, a2              /* t2 = s1 col ptr */
        mv      t3, a4              /* t3 = s2 col ptr */
        mv      t4, a0              /* t4 = dst col ptr */
        sub     t5, a0, t0          /* t5 = dst_top col ptr (= dst - linesize_bytes) */
        mv      t6, a6              /* t6 = remaining pixels */
        mv      s0, zero            /* s0 = carry = 0 */

.L_ssd_col:
        /* Step 1: compute sq = (s1[x]-s2[x])^2 vectorized */
        vsetvli t0, t6, e8, m1, ta, ma     /* t0 = VL (≤32) */

        vle8.v  v2, (t2)            /* s1[x..x+vl-1] as u8 */
        vle8.v  v4, (t3)            /* s2[x..x+vl-1] as u8 */

        vsetvli zero, t0, e16, m2, ta, ma
        vzext.vf2 v12, v2           /* s1 → u16 in v12-v13  [m2, 12 is even ✓] */
        vzext.vf2 v14, v4           /* s2 → u16 in v14-v15  [m2, 14 is even ✓] */
        vsub.vv   v12, v12, v14     /* d = s1-s2 */
        vmul.vv   v12, v12, v12     /* sq = d*d (≤65025, fits u16) */

        /* Step 2: row_term = dst_top[x] - dst_top[x-1] + sq */
        vsetvli zero, t0, e32, m4, ta, ma
        /* v0-v3 = wzext.vf2 of v12-v13; source m2 group (12-13) ∉ dest m4 group (0-3) ✓ */
        vzext.vf2 v0, v12           /* sq → u32 in v0-v3    [m4, 0 is /4 ✓] */

        vle32.v   v4, (t5)          /* v4-v7  = dst_top[x..]  [m4, 4 is /4 ✓] */
        addi      t1, t5, -4        /* t1 = &dst_top[x-1] */
        vle32.v   v8, (t1)          /* v8-v11 = dst_top[x-1..] [m4, 8 is /4 ✓] */
        vsub.vv   v4, v4, v8        /* dst_top[x] - dst_top[x-1] */
        vadd.vv   v4, v4, v0        /* row_term = (top diff) + sq */

        /* Step 3: scalar carry prefix sum */
        addi    sp, sp, -256
        vse32.v  v4, (sp)           /* spill row_term (v4-v7, m4, /4 ✓) */

        /* Uses: t0=temp value (clobbers VL), t1=counter(=VL), s1=buf ptr */
        mv      t1, t0              /* t1 = count = VL */
        mv      s1, sp              /* s1 = ptr into spilled buffer */
.L_carry_loop:
        lw      t0, 0(s1)           /* t0 = row_term[i] */
        add     t0, t0, s0          /* += carry */
        sw      t0, 0(s1)           /* write dst[i] */
        mv      s0, t0              /* carry = dst[i] */
        addi    s1, s1, 4           /* next element */
        addi    t1, t1, -1
        bnez    t1, .L_carry_loop

        /* Re-query VL: same result since t6 unchanged and VLMAX_e8m1=VLMAX_e32m4=32 */
        vsetvli t0, t6, e32, m4, ta, ma

        vle32.v  v4, (sp)           /* reload prefix-summed dst values (m4, /4 ✓) */
        vse32.v  v4, (t4)           /* store to dst */
        addi    sp, sp, 256

        /* Step 4: advance all col pointers by VL */
        add     t2, t2, t0          /* s1_col += vl (bytes) */
        add     t3, t3, t0          /* s2_col += vl (bytes) */
        slli    t1, t0, 2           /* t1 = vl * 4 (u32 stride) */
        add     t4, t4, t1          /* dst_col += vl */
        add     t5, t5, t1          /* dst_top_col += vl */
        sub     t6, t6, t0          /* remaining -= vl */
        bnez    t6, .L_ssd_col

        /* Advance to next row */
        slli    t0, a1, 2           /* dst_linesize_bytes */
        add     a0, a0, t0          /* dst row += linesize */
        add     a2, a2, a3          /* s1 row += linesize1 */
        add     a4, a4, a5          /* s2 row += linesize2 */
        addi    a7, a7, -1
        j       .L_ssd_row

.L_ssd_done:
        ld      s0,  0(sp)
        ld      s1,  8(sp)
        addi    sp, sp, 16
        ret
endfunc

/*
 * void ff_compute_weights_line_rvv(
 *     const uint32_t *iia,           // a0
 *     const uint32_t *iib,           // a1
 *     const uint32_t *iid,           // a2
 *     const uint32_t *iie,           // a3
 *     const uint8_t  *src,           // a4
 *     float *total_weight,           // a5
 *     float *sum,                    // a6
 *     const float *weight_lut,       // a7
 *     int max_meaningful_diff,       // sp+0  (9th arg, int)
 *     int startx,                    // sp+8  (10th arg, int)
 *     int endx)                      // sp+16 (11th arg, int)
 *
 * Algorithm (32 pixels/iter, e32m4):
 *   diff = FFMIN(iie[x]-iid[x]-iib[x]+iia[x], max)
 *   weight = weight_lut[diff]           (vluxei32.v gather using diff*4 byte offset)
 *   total_weight[x] += weight
 *   sum[x] += weight * (float)src[x]
 *
 * Register map (e32m4, 32 elements/group):
 *   v0-v3:   diff = e-d-b+a (u32)
 *   v4-v7:   temp u32 loads
 *   v8-v11:  weight (f32, from lut gather)
 *   v12-v15: src as f32
 *   v16-v19: total_weight accumulator (f32)
 *   v20-v23: sum accumulator (f32)
 *   v24:     src bytes (e8m1)
 */
func ff_compute_weights_line_rvv, zve32f, zba
        lpad    0
        /* Load stack-passed args (no prologue, sp points directly to them).
         * All three are ptrdiff_t (64-bit on rv64). Use ld. */
        ld      t0, 0(sp)           /* t0 = max_meaningful_diff (ptrdiff_t) */
        ld      t1, 8(sp)           /* t1 = startx (ptrdiff_t) */
        ld      t2, 16(sp)          /* t2 = endx (ptrdiff_t) */

        /* Apply startx offset to all pointer args */
        slli    t3, t1, 2           /* t3 = startx * 4 (byte offset for u32/f32 arrays) */
        add     a0, a0, t3          /* iia += startx */
        add     a1, a1, t3          /* iib += startx */
        add     a2, a2, t3          /* iid += startx */
        add     a3, a3, t3          /* iie += startx */
        add     a5, a5, t3          /* total_weight += startx */
        add     a6, a6, t3          /* sum += startx */
        add     a4, a4, t1          /* src += startx (1-byte elements) */

        sub     t2, t2, t1          /* t2 = remaining = endx - startx */

.L_wl_loop:
        vsetvli t3, t2, e32, m4, ta, ma    /* t3 = VL (≤32) */
        beqz    t3, .L_wl_done

        /* diff = iie[x] - iid[x] - iib[x] + iia[x] */
        vle32.v v0, (a0)            /* v0 = iia */
        vle32.v v4, (a1)            /* v4 = iib */
        vsub.vv v0, v0, v4          /* v0 = iia - iib */
        vle32.v v4, (a2)            /* v4 = iid */
        vsub.vv v0, v0, v4          /* v0 = iia - iib - iid */
        vle32.v v4, (a3)            /* v4 = iie */
        vadd.vv v0, v4, v0          /* v0 = iie + (iia-iib-iid) = e-d-b+a */

        /* Clamp diff to [0, max_meaningful_diff] */
        vminu.vx v0, v0, t0         /* v0 = FFMIN(diff, max) */

        /* Convert to byte offset into weight_lut: diff * sizeof(float) = diff << 2 */
        vsll.vi  v0, v0, 2          /* v0 = byte offsets */

        /* Gather float weights: vluxei32.v e32m4 data, e32m4 indices */
        vluxei32.v v8, (a7), v0     /* v8-v11 = weight_lut[diff] as f32 */

        /* Load src[x] as u8, widen u8 → u32 → f32 */
        vsetvli zero, t3, e8, m1, ta, ma
        vle8.v   v24, (a4)          /* v24 = src bytes */
        vsetvli zero, t3, e32, m4, ta, ma
        vzext.vf4 v12, v24          /* v12-v15 = u32 widened (v24 ∉ v12-v15 ✓) */
        vfcvt.f.xu.v v12, v12       /* v12-v15 = f32 */

        /* Accumulate: total_weight += weight, sum += weight * src_f32 */
        vle32.v  v16, (a5)          /* v16-v19 = total_weight[x..x+vl-1] */
        vle32.v  v20, (a6)          /* v20-v23 = sum[x..x+vl-1] */
        vfadd.vv v16, v16, v8       /* total_weight += weight */
        vfmacc.vv v20, v8, v12      /* sum += weight * src_f32 */
        vse32.v  v16, (a5)
        vse32.v  v20, (a6)

        /* Advance all pointers by VL */
        slli    t4, t3, 2           /* t4 = vl * 4 bytes */
        add     a0, a0, t4
        add     a1, a1, t4
        add     a2, a2, t4
        add     a3, a3, t4
        add     a4, a4, t3          /* src: byte ptr, advance by vl */
        add     a5, a5, t4
        add     a6, a6, t4

        sub     t2, t2, t3          /* remaining -= vl */
        bnez    t2, .L_wl_loop

.L_wl_done:
        ret
endfunc
