/*
 * RVV 1.0 yadif filter_line (8-bit) for SpacemiT K1 (VLEN=256)
 * Copyright (C) 2024 perise
 *
 * Processes 32 u8 pixels/iter using e8m1 loads + e16m2 arithmetic.
 * Full algorithm: spatial score CHECK(±1,±2), temporal diff, b/f adjust, clamp.
 *
 * Reg map (e16m2 = 2 regs):
 *   v2-v3:   c  (cur[mrefs] as u16)
 *   v4-v5:   e  (cur[prefs] as u16)
 *   v6-v7:   d  (temporal avg, u16)
 *   v8-v9:   diff (u16, may become signed after b/f)
 *   v10-v11: spatial_pred (u16, updated per CHECK)
 *   v12-v13: spatial_score or temp
 *   v14-v15: temp_a
 *   v16-v17: temp_b
 *   v18-v19: temp_c
 *   v20-v21: temp_d
 *   v24-v25: temp_e
 *   v22, v23, v26: e8m1 load buffers
 *   v0-v1:   mask (v0 used for vmsltu/vmerge; v0-v1 allocated but only v0 used as mask)
 *
 * void ff_yadif_filter_line_8_rvv(
 *     void *dst, void *prev, void *cur, void *next,  // a0-a3
 *     int w, int prefs, int mrefs, int parity,        // a4-a7
 *     int mode);                                       // sp+0
 */

#include "libavutil/riscv/asm.S"

/* ABSDIFF8_TO16 vdst_e16m2, ptr1, ptr2
 * Loads u8 from ptr1 and ptr2, computes absdiff, widens to e16m2 in vdst.
 * Clobbers: v22, v23, v26 (e8m1); uses current vl (must be set in caller).
 */
.macro ABSDIFF8_TO16 vdst, ptr1, ptr2
        vsetvli zero, t5, e8, m1, ta, ma
        vle8.v  v22, (\ptr1)
        vle8.v  v23, (\ptr2)
        vmaxu.vv v26, v22, v23
        vminu.vv v22, v22, v23
        vsub.vv  v22, v26, v22
        vsetvli zero, t5, e16, m2, ta, ma
        vzext.vf2 \vdst, v22
.endm

func ff_yadif_filter_line_8_rvv, zve32x, zba
        lpad    0
        addi    sp, sp, -64
        sd      s0,  0(sp); sd s1,  8(sp); sd s2, 16(sp); sd s3, 24(sp)
        sd      s4, 32(sp); sd s5, 40(sp); sd s6, 48(sp); sd ra, 56(sp)

        /* mode from caller stack (was sp+0 before prologue; now sp+64) */
        lw      s6, 64(sp)
        andi    s6, s6, 2               /* s6 = mode & 2 */

        /* prev2/next2 based on parity */
        mv      s0, a2; mv s1, a3      /* default: prev2=cur, next2=next */
        beqz    a7, 1f
        mv      s0, a1; mv s1, a2      /* parity=1: prev2=prev, next2=cur */
1:
        add     s2, a2, a6              /* s2 = cur+mrefs */
        add     s3, a2, a5              /* s3 = cur+prefs */
        add     s4, a1, a6              /* s4 = prev+mrefs */
        add     s5, a1, a5              /* s5 = prev+prefs */
        mv      t6, a4                  /* t6 = remaining */

.L_outer:
        vsetvli t5, t6, e16, m2, ta, ma
        beqz    t5, .L_done

        /* --- c = cur[mrefs], e = cur[prefs] --- */
        vsetvli zero, t5, e8, m1, ta, ma
        vle8.v  v22, (s2); vle8.v v23, (s3)
        vsetvli zero, t5, e16, m2, ta, ma
        vzext.vf2 v2, v22; vzext.vf2 v4, v23

        /* --- d = (prev2[0]+next2[0])>>1 --- */
        vsetvli zero, t5, e8, m1, ta, ma
        vle8.v  v22, (s0); vle8.v v23, (s1)
        vsetvli zero, t5, e16, m2, ta, ma
        vzext.vf2 v6, v22; vzext.vf2 v14, v23
        vadd.vv v6, v6, v14; vsrl.vi v6, v6, 1

        /* --- spatial_pred = (c+e)>>1 --- */
        vadd.vv v10, v2, v4; vsrl.vi v10, v10, 1

        /* --- initial spatial_score --- */
        addi t0, s2, -1; addi t1, s3, -1
        ABSDIFF8_TO16 v12, t0, t1
        ABSDIFF8_TO16 v14, s2, s3
        vadd.vv v12, v12, v14
        addi t0, s2, 1; addi t1, s3, 1
        ABSDIFF8_TO16 v14, t0, t1
        vadd.vv v12, v12, v14
        vadd.vi v12, v12, -1

        /* --- diff = max3(td0>>1, td1, td2) --- */
        ABSDIFF8_TO16 v8, s0, s1
        vsrl.vi v8, v8, 1

        ABSDIFF8_TO16 v14, s4, s2
        ABSDIFF8_TO16 v16, s5, s3
        vadd.vv v14, v14, v16; vsrl.vi v14, v14, 1
        vmaxu.vv v8, v8, v14

        add t0, a3, a6; add t1, a3, a5
        ABSDIFF8_TO16 v14, t0, s2
        ABSDIFF8_TO16 v16, t1, s3
        vadd.vv v14, v14, v16; vsrl.vi v14, v14, 1
        vmaxu.vv v8, v8, v14

        /* --- CHECK(-1): score=|cm2-p0|+|cm1-pp1|+|c-pp2|, pred=(cm1+pp1)>>1 --- */
        addi t0, s2, -2
        ABSDIFF8_TO16 v16, t0, s3
        addi t0, s2, -1; addi t1, s3, 1
        ABSDIFF8_TO16 v18, t0, t1
        vadd.vv v16, v16, v18
        addi t1, s3, 2
        ABSDIFF8_TO16 v18, s2, t1
        vadd.vv v16, v16, v18
        addi t0, s2, -1; addi t1, s3, 1
        vsetvli zero, t5, e8, m1, ta, ma
        vle8.v v22, (t0); vle8.v v23, (t1)
        vsetvli zero, t5, e16, m2, ta, ma
        vzext.vf2 v18, v22; vzext.vf2 v20, v23
        vadd.vv v18, v18, v20; vsrl.vi v18, v18, 1
        vmsltu.vv v0, v16, v12
        vmerge.vvm v12, v12, v16, v0
        vmerge.vvm v10, v10, v18, v0

        /* --- CHECK(-2): score=|cm3-pp1|+|cm2-pp2|+|cm1-pp3|, pred=(cm2+pp2)>>1 --- */
        addi t0, s2, -3; addi t1, s3, 1
        ABSDIFF8_TO16 v16, t0, t1
        addi t0, s2, -2; addi t1, s3, 2
        ABSDIFF8_TO16 v18, t0, t1
        vadd.vv v16, v16, v18
        addi t0, s2, -1; addi t1, s3, 3
        ABSDIFF8_TO16 v18, t0, t1
        vadd.vv v16, v16, v18
        addi t0, s2, -2; addi t1, s3, 2
        vsetvli zero, t5, e8, m1, ta, ma
        vle8.v v22, (t0); vle8.v v23, (t1)
        vsetvli zero, t5, e16, m2, ta, ma
        vzext.vf2 v18, v22; vzext.vf2 v20, v23
        vadd.vv v18, v18, v20; vsrl.vi v18, v18, 1
        vmsltu.vv v0, v16, v12
        vmerge.vvm v12, v12, v16, v0
        vmerge.vvm v10, v10, v18, v0

        /* --- CHECK(+1): score=|c-pm2|+|cp1-pm1|+|cp2-p0|, pred=(cp1+pm1)>>1 --- */
        addi t1, s3, -2
        ABSDIFF8_TO16 v16, s2, t1
        addi t0, s2, 1; addi t1, s3, -1
        ABSDIFF8_TO16 v18, t0, t1
        vadd.vv v16, v16, v18
        addi t0, s2, 2
        ABSDIFF8_TO16 v18, t0, s3
        vadd.vv v16, v16, v18
        addi t0, s2, 1; addi t1, s3, -1
        vsetvli zero, t5, e8, m1, ta, ma
        vle8.v v22, (t0); vle8.v v23, (t1)
        vsetvli zero, t5, e16, m2, ta, ma
        vzext.vf2 v18, v22; vzext.vf2 v20, v23
        vadd.vv v18, v18, v20; vsrl.vi v18, v18, 1
        vmsltu.vv v0, v16, v12
        vmerge.vvm v12, v12, v16, v0
        vmerge.vvm v10, v10, v18, v0

        /* --- CHECK(+2): score=|cp1-pm3|+|cp2-pm2|+|cp3-pm1|, pred=(cp2+pm2)>>1 --- */
        addi t0, s2, 1; addi t1, s3, -3
        ABSDIFF8_TO16 v16, t0, t1
        addi t0, s2, 2; addi t1, s3, -2
        ABSDIFF8_TO16 v18, t0, t1
        vadd.vv v16, v16, v18
        addi t0, s2, 3; addi t1, s3, -1
        ABSDIFF8_TO16 v18, t0, t1
        vadd.vv v16, v16, v18
        addi t0, s2, 2; addi t1, s3, -2
        vsetvli zero, t5, e8, m1, ta, ma
        vle8.v v22, (t0); vle8.v v23, (t1)
        vsetvli zero, t5, e16, m2, ta, ma
        vzext.vf2 v18, v22; vzext.vf2 v20, v23
        vadd.vv v18, v18, v20; vsrl.vi v18, v18, 1
        vmsltu.vv v0, v16, v12
        vmerge.vvm v12, v12, v16, v0
        vmerge.vvm v10, v10, v18, v0    /* v10 = final spatial_pred */

        /* --- b/f secondary adjustment (skip if mode & 2) --- */
        bnez    s6, .L_skip_bf

        slli    t2, a6, 1               /* 2*mrefs */
        slli    t3, a5, 1               /* 2*prefs */

        /* b = (prev2[2*mrefs]+next2[2*mrefs])>>1 */
        add t0, s0, t2; add t1, s1, t2
        vsetvli zero, t5, e8, m1, ta, ma
        vle8.v v22, (t0); vle8.v v23, (t1)
        vsetvli zero, t5, e16, m2, ta, ma
        vzext.vf2 v12, v22; vzext.vf2 v14, v23
        vadd.vv v12, v12, v14; vsrl.vi v12, v12, 1   /* v12 = b */

        /* f = (prev2[2*prefs]+next2[2*prefs])>>1 */
        add t0, s0, t3; add t1, s1, t3
        vsetvli zero, t5, e8, m1, ta, ma
        vle8.v v22, (t0); vle8.v v23, (t1)
        vsetvli zero, t5, e16, m2, ta, ma
        vzext.vf2 v14, v22; vzext.vf2 v16, v23
        vadd.vv v14, v14, v16; vsrl.vi v14, v14, 1   /* v14 = f */

        /* Signed diffs: v12=b-c, v14=f-e, v16=d-e, v18=d-c */
        vsub.vv v12, v12, v2            /* b-c */
        vsub.vv v14, v14, v4            /* f-e */
        vsub.vv v16, v6, v4             /* d-e */
        vsub.vv v18, v6, v2             /* d-c */

        /* max_adj = max3(d-e, d-c, min(b-c,f-e)) */
        vmin.vv v20, v12, v14           /* min(b-c, f-e)     → v20 */
        vmax.vv v24, v16, v18           /* max(d-e, d-c)      → v24 */
        vmax.vv v24, v24, v20           /* v24 = max_adj */

        /* min_adj = min3(d-e, d-c, max(b-c,f-e)) */
        vmax.vv v20, v12, v14           /* max(b-c, f-e)      → v20 */
        vmin.vv v16, v16, v18           /* min(d-e, d-c)      → v16 */
        vmin.vv v16, v16, v20           /* v16 = min_adj */

        /* neg_max_adj = -max_adj */
        vrsub.vi v20, v24, 0            /* v20 = -max_adj */

        /* diff = max3(diff, min_adj, neg_max_adj) -- signed */
        vmax.vv v8, v8, v16
        vmax.vv v8, v8, v20

.L_skip_bf:
        /* --- Clamp spatial_pred to [d-diff, d+diff] --- */
        vsub.vv v14, v6, v8             /* lo = d-diff */
        vadd.vv v16, v6, v8             /* hi = d+diff */
        vmax.vv v10, v10, v14
        vmin.vv v10, v10, v16
        vmax.vx v10, v10, zero          /* floor at 0 (signed max with 0) */

        /* --- Narrow to u8 and store --- */
        vsetvli zero, t5, e8, m1, ta, ma
        vnclipu.wi v22, v10, 0          /* e16m2 → e8m1, sat [0,255] */
        vse8.v  v22, (a0)

        /* --- Advance all row pointers by vl --- */
        add a0, a0, t5; add a1, a1, t5
        add a2, a2, t5; add a3, a3, t5
        add s0, s0, t5; add s1, s1, t5
        add s2, s2, t5; add s3, s3, t5
        add s4, s4, t5; add s5, s5, t5
        sub t6, t6, t5
        bnez t6, .L_outer

.L_done:
        ld s0, 0(sp); ld s1, 8(sp); ld s2, 16(sp); ld s3, 24(sp)
        ld s4, 32(sp); ld s5, 40(sp); ld s6, 48(sp); ld ra, 56(sp)
        addi sp, sp, 64
        ret
endfunc
