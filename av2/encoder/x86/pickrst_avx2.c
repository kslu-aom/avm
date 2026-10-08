/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#include <immintrin.h>  // AVX2
#include "avm_dsp/x86/mem_sse2.h"
#include "avm_dsp/x86/synonyms.h"
#include "avm_dsp/x86/synonyms_avx2.h"
#include "avm_dsp/x86/transpose_sse2.h"

#include "config/av2_rtcd.h"
#include "av2/common/restoration.h"
#include "av2/encoder/pickrst.h"

// Number of extra zero feature columns so that groups of 4 features can be
// processed without a bounds check.
#define FEAT_PAD 3

// Returns { sum(a), sum(b), sum(c), sum(d) } of the 32-bit lanes.
static AVM_INLINE __m128i hadd4_s32x8(__m256i a, __m256i b, __m256i c,
                                      __m256i d) {
  const __m256i ab = _mm256_hadd_epi32(a, b);
  const __m256i cd = _mm256_hadd_epi32(c, d);
  const __m256i abcd = _mm256_hadd_epi32(ab, cd);
  return _mm_add_epi32(_mm256_castsi256_si128(abcd),
                       _mm256_extracti128_si256(abcd, 1));
}

// dst[0..3] += v[0..3]. The stats are integers below 2^53, so the double
// additions are exact.
static AVM_INLINE void add_s32x4_to_f64(double *dst, __m128i v) {
  _mm256_storeu_pd(dst,
                   _mm256_add_pd(_mm256_loadu_pd(dst), _mm256_cvtepi32_pd(v)));
}

// dst[0..n-1] += v[0..n-1], n < 4.
static AVM_INLINE void add_s32xn_to_f64(double *dst, __m128i v, int n) {
  int32_t tmp[4];
  _mm_storeu_si128((__m128i *)tmp, v);
  for (int i = 0; i < n; ++i) dst[i] += tmp[i];
}

static AVM_INLINE __m256i dot32(const __m256i *v0, const __m256i *v1, int k,
                                int l) {
  return _mm256_add_epi32(_mm256_madd_epi16(v0[k], v0[l]),
                          _mm256_madd_epi16(v1[k], v1[l]));
}

void av2_accumulate_wienerns_correlation_batch_avx2(
    double *A_base, double *b_base, const int16_t *buf_cols,
    const int16_t *y_arr, int batch_size, int num_feat) {
  (void)batch_size;
  // Columns [0, num_feat) are the features, column num_feat is y and the
  // remaining columns are zero padding.
  __m256i v0[WIENERNS_TAPS_MAX + 1 + FEAT_PAD];
  __m256i v1[WIENERNS_TAPS_MAX + 1 + FEAT_PAD];
  for (int k = 0; k < num_feat; ++k) {
    const int16_t *col_k = buf_cols + k * WIENERNS_STATS_BATCH_SIZE;
    v0[k] = _mm256_loadu_si256((const __m256i *)col_k);
    v1[k] = _mm256_loadu_si256((const __m256i *)(col_k + 16));
  }
  const int y_idx = num_feat;
  v0[y_idx] = _mm256_loadu_si256((const __m256i *)y_arr);
  v1[y_idx] = _mm256_loadu_si256((const __m256i *)(y_arr + 16));
  for (int k = num_feat + 1; k < num_feat + 1 + FEAT_PAD; ++k) {
    v0[k] = _mm256_setzero_si256();
    v1[k] = _mm256_setzero_si256();
  }

  // b[k] += <col_k, y>.
  int k = 0;
  for (; k + 4 <= num_feat; k += 4) {
    const __m128i s =
        hadd4_s32x8(dot32(v0, v1, k, y_idx), dot32(v0, v1, k + 1, y_idx),
                    dot32(v0, v1, k + 2, y_idx), dot32(v0, v1, k + 3, y_idx));
    add_s32x4_to_f64(b_base + k, s);
  }
  if (k < num_feat) {
    // Columns past num_feat are y or zero; the extra sums are discarded.
    const __m128i s =
        hadd4_s32x8(dot32(v0, v1, k, y_idx), dot32(v0, v1, k + 1, y_idx),
                    dot32(v0, v1, k + 2, y_idx), dot32(v0, v1, k + 3, y_idx));
    add_s32xn_to_f64(b_base + k, s, num_feat - k);
  }

  // A[k][l] += <col_k, col_l>, l <= k.
  for (k = 0; k < num_feat; ++k) {
    double *const A_row = A_base + k * num_feat;
    int l = 0;
    for (; l + 4 <= k + 1; l += 4) {
      const __m128i s =
          hadd4_s32x8(dot32(v0, v1, k, l), dot32(v0, v1, k, l + 1),
                      dot32(v0, v1, k, l + 2), dot32(v0, v1, k, l + 3));
      add_s32x4_to_f64(A_row + l, s);
    }
    if (l <= k) {
      // l + 3 <= k + 3 <= num_feat + 2, which is within the padded columns.
      const __m128i s =
          hadd4_s32x8(dot32(v0, v1, k, l), dot32(v0, v1, k, l + 1),
                      dot32(v0, v1, k, l + 2), dot32(v0, v1, k, l + 3));
      add_s32xn_to_f64(A_row + l, s, k + 1 - l);
    }
  }
}

int64_t av2_compute_wienerns_features_avx2(const uint16_t *dgd,
                                           const uint16_t *luma,
                                           const uint16_t *src, int width,
                                           const WienernsFeatureTaps *taps,
                                           int16_t *feat, int16_t *y) {
  assert(width <= WIENERNS_STATS_BATCH_SIZE);
  const int num_feat = taps->num_feat;
  // When there is no luma there are no luma taps either; point at dgd so that
  // the (unused) center load below is valid.
  const uint16_t *lbase = luma ? luma : dgd;
  // Each 32-bit lane accumulates at most 4 squares of 13-bit values.
  __m256i sse = _mm256_setzero_si256();
  int p = 0;
  for (; p + 16 <= width; p += 16) {
    const uint16_t *d = dgd + p;
    const uint16_t *lu = lbase + p;
    const __m256i dc = _mm256_loadu_si256((const __m256i *)d);
    const __m256i lc = _mm256_loadu_si256((const __m256i *)lu);
    for (int k = 0; k < num_feat; ++k) {
      // sum(x[t] - x[0]) = sum(x[t]) - n * x[0] in 16-bit wrap-around
      // arithmetic.
      const int d0 = taps->dgd_start[k], d1 = taps->dgd_start[k + 1];
      const int l0 = taps->luma_start[k], l1 = taps->luma_start[k + 1];
      __m256i acc = _mm256_set1_epi16(taps->feat_const[k]);
      for (int t = d0; t < d1; ++t) {
        acc = _mm256_add_epi16(
            acc,
            _mm256_loadu_si256((const __m256i *)(d + taps->dgd_offset[t])));
      }
      for (int t = l0; t < l1; ++t) {
        acc = _mm256_add_epi16(
            acc,
            _mm256_loadu_si256((const __m256i *)(lu + taps->luma_offset[t])));
      }
      acc = _mm256_sub_epi16(
          acc, _mm256_mullo_epi16(dc, _mm256_set1_epi16((int16_t)(d1 - d0))));
      acc = _mm256_sub_epi16(
          acc, _mm256_mullo_epi16(lc, _mm256_set1_epi16((int16_t)(l1 - l0))));
      _mm256_storeu_si256((__m256i *)(feat + k * WIENERNS_STATS_BATCH_SIZE + p),
                          acc);
    }
    const __m256i yv =
        _mm256_sub_epi16(_mm256_loadu_si256((const __m256i *)(src + p)), dc);
    _mm256_storeu_si256((__m256i *)(y + p), yv);
    sse = _mm256_add_epi32(sse, _mm256_madd_epi16(yv, yv));
  }
  __m128i sse8 = _mm_setzero_si128();
  for (; p + 8 <= width; p += 8) {
    const uint16_t *d = dgd + p;
    const uint16_t *lu = lbase + p;
    const __m128i dc = _mm_loadu_si128((const __m128i *)d);
    const __m128i lc = _mm_loadu_si128((const __m128i *)lu);
    for (int k = 0; k < num_feat; ++k) {
      const int d0 = taps->dgd_start[k], d1 = taps->dgd_start[k + 1];
      const int l0 = taps->luma_start[k], l1 = taps->luma_start[k + 1];
      __m128i acc = _mm_set1_epi16(taps->feat_const[k]);
      for (int t = d0; t < d1; ++t) {
        acc = _mm_add_epi16(
            acc, _mm_loadu_si128((const __m128i *)(d + taps->dgd_offset[t])));
      }
      for (int t = l0; t < l1; ++t) {
        acc = _mm_add_epi16(
            acc, _mm_loadu_si128((const __m128i *)(lu + taps->luma_offset[t])));
      }
      acc = _mm_sub_epi16(
          acc, _mm_mullo_epi16(dc, _mm_set1_epi16((int16_t)(d1 - d0))));
      acc = _mm_sub_epi16(
          acc, _mm_mullo_epi16(lc, _mm_set1_epi16((int16_t)(l1 - l0))));
      _mm_storeu_si128((__m128i *)(feat + k * WIENERNS_STATS_BATCH_SIZE + p),
                       acc);
    }
    const __m128i yv =
        _mm_sub_epi16(_mm_loadu_si128((const __m128i *)(src + p)), dc);
    _mm_storeu_si128((__m128i *)(y + p), yv);
    sse8 = _mm_add_epi32(sse8, _mm_madd_epi16(yv, yv));
  }
  const __m128i s4 =
      _mm_add_epi32(_mm_add_epi32(_mm256_castsi256_si128(sse),
                                  _mm256_extracti128_si256(sse, 1)),
                    sse8);
  const __m128i s2 = _mm_add_epi32(s4, _mm_srli_si128(s4, 8));
  const __m128i s1 = _mm_add_epi32(s2, _mm_srli_si128(s2, 4));
  int64_t total = (uint32_t)_mm_cvtsi128_si32(s1);
  if (p < width) {
    total += av2_compute_wienerns_features_c(dgd + p, luma ? luma + p : NULL,
                                             src + p, width - p, taps, feat + p,
                                             y + p);
  }
  return total;
}
