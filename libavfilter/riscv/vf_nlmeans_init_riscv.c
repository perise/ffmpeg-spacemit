/*
 * RVV nlmeans DSP init for SpacemiT K1
 * Copyright (C) 2024 perise
 *
 * This file is part of FFmpeg.
 * LGPL 2.1 or later.
 */

#include "config.h"
#include "libavutil/attributes.h"
#include "libavutil/riscv/cpu.h"
#include "libavfilter/vf_nlmeans.h"

void ff_compute_safe_ssd_integral_image_rvv(uint32_t *dst,
                                             ptrdiff_t dst_linesize_32,
                                             const uint8_t *s1, ptrdiff_t linesize1,
                                             const uint8_t *s2, ptrdiff_t linesize2,
                                             int w, int h);

void ff_compute_weights_line_rvv(const uint32_t *const iia,
                                  const uint32_t *const iib,
                                  const uint32_t *const iid,
                                  const uint32_t *const iie,
                                  const uint8_t *const src,
                                  float *total_weight,
                                  float *sum,
                                  const float *const weight_lut,
                                  int max_meaningful_diff,
                                  int startx, int endx);

av_cold void ff_nlmeans_init_riscv(NLMeansDSPContext *dsp)
{
#if HAVE_RVV
    int flags = av_get_cpu_flags();
    if ((flags & AV_CPU_FLAG_RVV_I32) && (flags & AV_CPU_FLAG_RVB)) {
        dsp->compute_safe_ssd_integral_image = ff_compute_safe_ssd_integral_image_rvv;
    }
    if ((flags & AV_CPU_FLAG_RVV_F32) && (flags & AV_CPU_FLAG_RVB)) {
        dsp->compute_weights_line = ff_compute_weights_line_rvv;
        
    }
#endif
}
