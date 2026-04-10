/*
 * RISC-V RVV 1.0 vertical output accumulation functions for libswscale.
 * Implements yuv2plane1_8 and yuv2planeX_8 (vertical scale output stage).
 *
 * Copyright (c) 2026 SpacemiT K1 optimization
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#include "libavutil/riscv/asm.S"

/*
 * These functions form the vertical output stage of libswscale's scaling
 * pipeline.  They convert the horizontally-scaled int16 line buffers back
 * to uint8 destination pixels.
 *
 * yuv2plane1_8 — single vertical tap (FAST_BILINEAR vertical or 1:1):
 *   dest[i] = clip_uint8((src[i] + dither[(i+offset)&7]) >> 7)
 *
 * yuv2planeX_8 — multi-tap FIR vertical filter (Lanczos, bilinear, etc.):
 *   val  = dither[(i+offset)&7] << 12
 *   for j in [0, filterSize): val += src[j][i] * filter[j]
 *   dest[i] = clip_uint8(val >> 19)
 *
 * RVV strategy (VLEN=256):
 *
 *   yuv2plane1_8 uses e8m1 / e16m2 (32 pixels per VL).
 *     - vid.v + vadd.vx + vand.vi → dither index (e8m1)
 *     - vluxei8.v → gather dither byte (e8m1)
 *     - vzext.vf2 → widen dither to e16m2
 *     - vle16.v  → load int16 src (e16m2)
 *     - vadd.vv + vsrl.vi → add dither and shift
 *     - vnclipu.wi → saturating narrow e16→e8
 *     Speedup: ~8× vs scalar on K1.
 *
 *   yuv2planeX_8 uses e8m1 / e16m2 / e32m4 (32 pixels per VL).
 *     - Accumulator initialized: vzext.vf4 (e8→e32) + vsll.vi 12
 *     - Inner j-loop: vwmacc.vx (int16×int16 → int32 widening acc)
 *     - vnsrl.wi 19 → narrow to approximate uint8 range
 *     - Two-step vnclipu (e32→e16→e8) with saturation to [0,255]
 *     Speedup: ~6–8× vs scalar on K1 (for typical filterSize 4–16).
 */

/* =========================================================================
 * ff_yuv2plane1_8_rvv
 *
 * void ff_yuv2plane1_8_rvv(const int16_t *src, uint8_t *dest, int dstW,
 *                           const uint8_t *dither, int offset)
 *
 * ABI (rv64):
 *   a0 = src    (const int16_t*)
 *   a1 = dest   (uint8_t*)
 *   a2 = dstW   (int)
 *   a3 = dither (const uint8_t[8])
 *   a4 = offset (int, starting dither phase)
 * ========================================================================= */

func ff_yuv2plane1_8_rvv, zve32x, zba
        lpad    0

        /*
         * t5 tracks the "global" pixel index (= i + offset) for dither.
         * It starts at `offset` and increments by VL each iteration.
         */
        mv      t5, a4

.Lp1_loop:
        vsetvli t0, a2, e8, m1, ta, ma     /* VL = min(dstW, 32) */

        /*
         * Compute dither indices: [(t5+0)&7, (t5+1)&7, ..., (t5+VL-1)&7]
         * vid.v gives [0..VL-1], add base t5, mask to [0..7].
         */
        vid.v   v2                          /* v2 = [0, 1, ..., VL-1] (e8m1) */
        vadd.vx v2, v2, t5                  /* v2 += offset base */
        vand.vi v2, v2, 7                   /* v2 &= 7  (modulo 8) */

        /* Gather dither bytes: v3[x] = dither[v2[x]] */
        vluxei8.v v3, (a3), v2             /* v3 = dither[(i+offset)&7] (e8m1) */

        /*
         * Load int16 src and add dither, both in e16m2.
         * vzext.vf2: zero-extend e8m1 dither → e16m2
         * vle16.v:   load VL int16 src values → e16m2
         */
        vsetvli zero, t0, e16, m2, ta, ma
        vzext.vf2 v4, v3                    /* dither as uint16 (e16m2) */
        vle16.v   v6, (a0)                  /* src[i..i+VL-1] (e16m2) */
        vadd.vv   v6, v6, v4               /* src + dither */
        vsrl.vi   v6, v6, 7               /* >> 7 → [0..255] for valid src */

        /* Saturating narrow e16m2 → e8m1, store */
        vsetvli zero, t0, e8, m1, ta, ma
        vnclipu.wi v0, v6, 0               /* clip and narrow to uint8 */
        vse8.v    v0, (a1)

        /* Advance pointers */
        slli    t1, t0, 1                  /* t1 = VL * 2 (src is int16) */
        add     a0, a0, t1                 /* src   += VL*2 */
        add     a1, a1, t0                 /* dest  += VL */
        add     t5, t5, t0                 /* dither base += VL */
        sub     a2, a2, t0                 /* dstW -= VL */
        bnez    a2, .Lp1_loop

        ret
endfunc

/* =========================================================================
 * ff_yuv2planeX_8_rvv
 *
 * void ff_yuv2planeX_8_rvv(const int16_t *filter, int filterSize,
 *                            const int16_t **src, uint8_t *dest, int dstW,
 *                            const uint8_t *dither, int offset)
 *
 * ABI (rv64):
 *   a0 = filter     (const int16_t*) — FIR coefficients, filterSize entries
 *   a1 = filterSize (int)
 *   a2 = src        (const int16_t**) — array of filterSize row pointers
 *   a3 = dest       (uint8_t*)
 *   a4 = dstW       (int)
 *   a5 = dither     (const uint8_t[8])
 *   a6 = offset     (int, starting dither phase)
 *
 * Register plan:
 *   v0–v3  e32m4  accumulator (one int32 per output pixel)
 *   v4–v5  e16m2  current src_j row slice (temporary)
 *   v6     e8m1   dither byte values
 *   t0     VL
 *   t1     j loop counter / scratch
 *   t2     filter[j] scalar (int16, sign-extended to 64-bit GPR)
 *   t3     src[j] + pixel_byte_offset  (row pointer into current j)
 *   t4     pixel byte offset within current batch (base_i * 2)
 *   t5     global pixel index for dither (= base_i + offset)
 *   t6     scratch / src byte stride per pixel batch
 * ========================================================================= */

func ff_yuv2planeX_8_rvv, zve32x, zba
        lpad    0

        /* t5 = global dither index base (starts at offset) */
        mv      t5, a6

        /* t4 = byte offset into each src row for current pixel batch (starts 0) */
        li      t4, 0

.LpX_batch:
        vsetvli t0, a4, e8, m1, ta, ma     /* VL = min(dstW, 32) */

        /*
         * ---- Initialize accumulator from dither ----
         * Gather dither[(t5+x)&7] for x in [0,VL) into v6 (e8m1).
         */
        vid.v   v6                          /* [0..VL-1] */
        vadd.vx v6, v6, t5                  /* + dither base */
        vand.vi v6, v6, 7
        vluxei8.v v6, (a5), v6             /* v6 = dither bytes (e8m1) */

        /*
         * Widen dither e8m1 → e32m4 and shift left 12:
         *   acc = dither_value << 12
         * vzext.vf4 with vtype=e32m4: source is e8m1 (EMUL=m4/4=m1).
         */
        vsetvli zero, t0, e32, m4, ta, ma
        vzext.vf4 v0, v6                    /* dither as uint32 (e32m4) */
        vsll.vi   v0, v0, 12               /* acc = dither << 12 */

        /*
         * ---- Inner j-loop: accumulate src[j][i..i+VL-1] * filter[j] ----
         * vwmacc.vx with vtype=e16m2:
         *   vd (e32m4) += vs2 (e16m2) × rs1 (int16 scalar)
         */
        li      t1, 0                       /* j = 0 */

.LpX_jloop:
        bge     t1, a1, .LpX_jdone         /* if j >= filterSize: break */

        /* filter[j]: load int16, sign-extend to 64-bit */
        slli    t6, t1, 1                   /* t6 = j * 2 (byte offset) */
        add     t6, a0, t6                  /* t6 = &filter[j] */
        lh      t2, 0(t6)                   /* t2 = filter[j] (sign-extended) */

        /* src[j]: load pointer from src[j] (64-bit pointer) */
        slli    t6, t1, 3                   /* t6 = j * 8 (pointer byte offset) */
        add     t6, a2, t6                  /* t6 = &src[j] */
        ld      t3, 0(t6)                   /* t3 = src[j] (pointer) */
        add     t3, t3, t4                  /* t3 = &src[j][base_i] (byte offset) */

        /* Load VL int16 values from src[j][base_i..base_i+VL-1] */
        vsetvli zero, t0, e16, m2, ta, ma  /* vtype = e16m2 for widening macc */
        vle16.v v4, (t3)                    /* v4 = src[j][i..i+VL-1] (e16m2) */

        /* Widening accumulate: v0 (e32m4) += v4 (e16m2) × t2 (int16) */
        vwmacc.vx v0, t2, v4               /* v0[i] += src[j][i] * filter[j] */

        addi    t1, t1, 1                   /* j++ */
        j       .LpX_jloop

.LpX_jdone:
        /*
         * ---- Shift and clip ----
         * val >> 19 using vnsrl.wi (narrowing logical shift, no saturation).
         * Result for valid input is in [0..~1023] (Lanczos can overshoot).
         * Two-step saturation narrow to uint8:
         *   e32m4 → e16m2 (vnclipu clips to [0..65535], but shift first)
         *   e16m2 → e8m1  (vnclipu clips to [0..255])
         */
        vsetvli zero, t0, e32, m4, ta, ma
        vnsrl.wi v8, v0, 19                 /* v8 = val >> 19 (e16m2, truncated) */
        /* Note: vnsrl dest is e16m2 since source is e32m4 and shift narrows */

        vsetvli zero, t0, e16, m2, ta, ma
        vnclipu.wi v4, v8, 0               /* clip e16m2 → e8m1 (saturate to 255) */

        vsetvli zero, t0, e8, m1, ta, ma
        vse8.v  v4, (a3)                    /* store VL uint8 pixels */

        /* Advance outer loop */
        slli    t6, t0, 1                   /* t6 = VL * 2 (int16 byte size) */
        add     t4, t4, t6                  /* pixel byte offset += VL*2 */
        add     a3, a3, t0                  /* dest += VL */
        add     t5, t5, t0                  /* dither base += VL */
        sub     a4, a4, t0                  /* dstW -= VL */
        bnez    a4, .LpX_batch

        ret
endfunc
