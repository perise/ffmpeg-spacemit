/*
 * RVV image utility functions for SpacemiT K1 (VLEN=256)
 * Copyright (C) 2024 perise
 *
 * This file is part of FFmpeg.
 * LGPL 2.1 or later.
 *
 * ff_image_copy_plane_rvv:
 *   Bulk row-by-row plane copy using e8m8 (LMUL=8).
 *   With VLEN=256: VLMAX = 256/8 * 8 = 256 bytes/iter.
 *   This covers a full 1920-pixel luma row in ~8 iterations.
 *
 *   Prototype:
 *     void ff_image_copy_plane_rvv(uint8_t *dst, ptrdiff_t dst_linesize,
 *                                   const uint8_t *src, ptrdiff_t src_linesize,
 *                                   ptrdiff_t bytewidth, int height);
 *   Arguments:
 *     a0 = dst
 *     a1 = dst_linesize
 *     a2 = src
 *     a3 = src_linesize
 *     a4 = bytewidth
 *     a5 = height
 */

#include "libavutil/riscv/asm.S"

func ff_image_copy_plane_rvv, zve32x
        lpad    0
        /* Early exit for degenerate inputs */
        beqz    a5, .L_cp_done
        beqz    a4, .L_cp_done

.L_cp_row:
        mv      t0, a4              /* t0 = bytes_remaining for this row */
        mv      t1, a0              /* t1 = dst_col (running dst pointer) */
        mv      t2, a2              /* t2 = src_col (running src pointer) */

.L_cp_col:
        /* Use LMUL=8 for maximum 256 bytes per iteration */
        vsetvli t3, t0, e8, m8, ta, ma
        vle8.v  v0, (t2)
        vse8.v  v0, (t1)
        add     t1, t1, t3
        add     t2, t2, t3
        sub     t0, t0, t3
        bnez    t0, .L_cp_col

        /* Advance to next row */
        add     a0, a0, a1          /* dst += dst_linesize */
        add     a2, a2, a3          /* src += src_linesize */
        addi    a5, a5, -1
        bnez    a5, .L_cp_row

.L_cp_done:
        ret
endfunc
