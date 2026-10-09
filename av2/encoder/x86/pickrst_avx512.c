/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#include <immintrin.h>

#include "config/av2_rtcd.h"
#include "av2/common/restoration.h"
#include "av2/encoder/pickrst.h"

static AVM_INLINE int64_t horizontal_add_s32x16(__m512i m) {
  return (int64_t)_mm512_reduce_add_epi32(m);
}

void av2_accumulate_wienerns_correlation_batch_avx512(
    int64_t *A_base, int64_t *b_base, const int16_t *buf_cols,
    const int16_t *y_arr, int batch_size, int num_feat) {
  (void)batch_size;
  const __m512i vy = _mm512_loadu_si512((const __m512i *)y_arr);

  __m512i vk[WIENERNS_TAPS_MAX];

  for (int k = 0; k < num_feat; ++k) {
    const int16_t *col_k = buf_cols + k * WIENERNS_STATS_BATCH_SIZE;
    vk[k] = _mm512_loadu_si512((const __m512i *)col_k);

    const __m512i mb = _mm512_madd_epi16(vk[k], vy);
    b_base[k] += horizontal_add_s32x16(mb);
  }

  for (int k = 0; k < num_feat; ++k) {
    int64_t *const A_row = A_base + k * num_feat;
    const __m512i v_k = vk[k];
    for (int l = 0; l <= k; ++l) {
      const __m512i m = _mm512_madd_epi16(v_k, vk[l]);
      A_row[l] += horizontal_add_s32x16(m);
    }
  }
}
