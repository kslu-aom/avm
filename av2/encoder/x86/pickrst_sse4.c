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

#include <smmintrin.h>
#include "avm_dsp/x86/synonyms.h"

#include "config/av2_rtcd.h"
#include "av2/common/restoration.h"
#include "av2/encoder/pickrst.h"

// Number of extra zero feature columns so that groups of 4 features can be
// processed without a bounds check.
#define FEAT_PAD 3

// Returns { sum(a), sum(b), sum(c), sum(d) } of the 32-bit lanes.
static AVM_INLINE __m128i hadd4_s32x4(__m128i a, __m128i b, __m128i c,
                                      __m128i d) {
  return _mm_hadd_epi32(_mm_hadd_epi32(a, b), _mm_hadd_epi32(c, d));
}

// dst[0..3] += sign-extended v[0..3].
static AVM_INLINE void add_s32x4_to_s64(int64_t *dst, __m128i v) {
  const __m128i d0 = _mm_loadu_si128((const __m128i *)dst);
  const __m128i d1 = _mm_loadu_si128((const __m128i *)(dst + 2));
  _mm_storeu_si128((__m128i *)dst, _mm_add_epi64(d0, _mm_cvtepi32_epi64(v)));
  _mm_storeu_si128((__m128i *)(dst + 2),
                   _mm_add_epi64(d1, _mm_cvtepi32_epi64(_mm_srli_si128(v, 8))));
}

// dst[0..n-1] += v[0..n-1], n < 4.
static AVM_INLINE void add_s32xn_to_s64(int64_t *dst, __m128i v, int n) {
  int32_t tmp[4];
  _mm_storeu_si128((__m128i *)tmp, v);
  for (int i = 0; i < n; ++i) dst[i] += tmp[i];
}

typedef struct {
  __m128i v[4];
} Col32;

static AVM_INLINE __m128i dot32(const Col32 *c, int k, int l) {
  const __m128i m0 = _mm_madd_epi16(c[k].v[0], c[l].v[0]);
  const __m128i m1 = _mm_madd_epi16(c[k].v[1], c[l].v[1]);
  const __m128i m2 = _mm_madd_epi16(c[k].v[2], c[l].v[2]);
  const __m128i m3 = _mm_madd_epi16(c[k].v[3], c[l].v[3]);
  return _mm_add_epi32(_mm_add_epi32(m0, m1), _mm_add_epi32(m2, m3));
}

static AVM_INLINE void load_col32(Col32 *c, const int16_t *src) {
  for (int i = 0; i < 4; ++i) {
    c->v[i] = _mm_loadu_si128((const __m128i *)(src + 8 * i));
  }
}

void av2_accumulate_wienerns_correlation_batch_sse4_1(
    int64_t *A_base, int64_t *b_base, const int16_t *buf_cols,
    const int16_t *y_arr, int batch_size, int num_feat) {
  (void)batch_size;
  // Columns [0, num_feat) are the features, column num_feat is y and the
  // remaining columns are zero padding.
  Col32 cols[WIENERNS_TAPS_MAX + 1 + FEAT_PAD];
  for (int k = 0; k < num_feat; ++k) {
    load_col32(&cols[k], buf_cols + k * WIENERNS_STATS_BATCH_SIZE);
  }
  const int y_idx = num_feat;
  load_col32(&cols[y_idx], y_arr);
  for (int k = num_feat + 1; k < num_feat + 1 + FEAT_PAD; ++k) {
    for (int i = 0; i < 4; ++i) cols[k].v[i] = _mm_setzero_si128();
  }

  // b[k] += <col_k, y>.
  int k = 0;
  for (; k + 4 <= num_feat; k += 4) {
    const __m128i s =
        hadd4_s32x4(dot32(cols, k, y_idx), dot32(cols, k + 1, y_idx),
                    dot32(cols, k + 2, y_idx), dot32(cols, k + 3, y_idx));
    add_s32x4_to_s64(b_base + k, s);
  }
  if (k < num_feat) {
    // Columns past num_feat are y or zero; the extra sums are discarded.
    const __m128i s =
        hadd4_s32x4(dot32(cols, k, y_idx), dot32(cols, k + 1, y_idx),
                    dot32(cols, k + 2, y_idx), dot32(cols, k + 3, y_idx));
    add_s32xn_to_s64(b_base + k, s, num_feat - k);
  }

  // A[k][l] += <col_k, col_l>, l <= k.
  for (k = 0; k < num_feat; ++k) {
    int64_t *const A_row = A_base + k * num_feat;
    int l = 0;
    for (; l + 4 <= k + 1; l += 4) {
      const __m128i s =
          hadd4_s32x4(dot32(cols, k, l), dot32(cols, k, l + 1),
                      dot32(cols, k, l + 2), dot32(cols, k, l + 3));
      add_s32x4_to_s64(A_row + l, s);
    }
    if (l <= k) {
      // l + 3 <= k + 3 <= num_feat + 2, which is within the padded columns.
      const __m128i s =
          hadd4_s32x4(dot32(cols, k, l), dot32(cols, k, l + 1),
                      dot32(cols, k, l + 2), dot32(cols, k, l + 3));
      add_s32xn_to_s64(A_row + l, s, k + 1 - l);
    }
  }
}

int64_t av2_compute_wienerns_features_sse4_1(const uint16_t *dgd,
                                             const uint16_t *luma,
                                             const uint16_t *src, int width,
                                             const WienernsFeatureTaps *taps,
                                             int16_t *feat, int16_t *y) {
  assert(width <= WIENERNS_STATS_BATCH_SIZE);
  const int num_feat = taps->num_feat;
  // When there is no luma there are no luma taps either; point at dgd so that
  // the (unused) center load below is valid.
  const uint16_t *lbase = luma ? luma : dgd;
  // Each 32-bit lane accumulates at most 8 squares of 13-bit values.
  __m128i sse = _mm_setzero_si128();
  int p = 0;
  for (; p + 8 <= width; p += 8) {
    const uint16_t *d = dgd + p;
    const uint16_t *lu = lbase + p;
    const __m128i dc = _mm_loadu_si128((const __m128i *)d);
    const __m128i lc = _mm_loadu_si128((const __m128i *)lu);
    for (int k = 0; k < num_feat; ++k) {
      // sum(x[t] - x[0]) = sum(x[t]) - n * x[0] in 16-bit wrap-around
      // arithmetic.
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
    sse = _mm_add_epi32(sse, _mm_madd_epi16(yv, yv));
  }
  const __m128i s2 = _mm_add_epi32(sse, _mm_srli_si128(sse, 8));
  const __m128i s1 = _mm_add_epi32(s2, _mm_srli_si128(s2, 4));
  int64_t total = (uint32_t)_mm_cvtsi128_si32(s1);
  if (p < width) {
    total += av2_compute_wienerns_features_c(dgd + p, luma ? luma + p : NULL,
                                             src + p, width - p, taps, feat + p,
                                             y + p);
  }
  return total;
}
