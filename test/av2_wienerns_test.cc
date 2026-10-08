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

#include <cmath>
#include <tuple>

#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

#include "test/acm_random.h"
#include "test/clear_system_state.h"
#include "test/register_state_check.h"
#include "test/util.h"

#include "config/avm_config.h"
#include "config/av2_rtcd.h"

#include "av2/common/restoration.h"
#include "av2/encoder/pickrst.h"

using libavm_test::ACMRandom;

namespace {

typedef void (*AccumulateWienernsCorrFunc)(double *A_base, double *b_base,
                                           const int16_t *buf_batch,
                                           const int16_t *y_batch,
                                           int batch_size, int num_feat);

typedef std::tuple<AccumulateWienernsCorrFunc> AccumulateWienernsCorrParam;

class WienernsCorrTest
    : public ::testing::TestWithParam<AccumulateWienernsCorrParam> {
 public:
  virtual void SetUp() {
    target_func_ = GET_PARAM(0);
    rnd_.Reset(ACMRandom::DeterministicSeed());
  }

  virtual void TearDown() { libavm_test::ClearSystemState(); }

  void RunRandomTest();
  void RunExtremeTest();
  void RunSpeedTest(int run_times);

 private:
  AccumulateWienernsCorrFunc target_func_;
  ACMRandom rnd_;
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(WienernsCorrTest);

static const int kTestFeatCounts[] = {
  1, 2, 3, 4, 5, 8, 9, 13, 16, 20, 25, 32
};
static const int kNumFeatTests =
    sizeof(kTestFeatCounts) / sizeof(kTestFeatCounts[0]);

void WienernsCorrTest::RunRandomTest() {
  double A_ref[WIENERNS_TAPS_MAX * WIENERNS_TAPS_MAX];
  double A_tst[WIENERNS_TAPS_MAX * WIENERNS_TAPS_MAX];
  double b_ref[WIENERNS_TAPS_MAX];
  double b_tst[WIENERNS_TAPS_MAX];
  int16_t buf[WIENERNS_TAPS_MAX * WIENERNS_STATS_BATCH_SIZE];
  int16_t y[WIENERNS_STATS_BATCH_SIZE];

  for (int f_idx = 0; f_idx < kNumFeatTests; ++f_idx) {
    const int num_feat = kTestFeatCounts[f_idx];
    if (num_feat > WIENERNS_TAPS_MAX) continue;

    for (int iter = 0; iter < 500 && !HasFatalFailure(); ++iter) {
      for (int i = 0; i < num_feat * num_feat; ++i) {
        const double init_val = static_cast<double>(rnd_.Rand31() % 100000);
        A_ref[i] = init_val;
        A_tst[i] = init_val;
      }
      for (int i = 0; i < num_feat; ++i) {
        const double init_val = static_cast<double>(rnd_.Rand31() % 100000);
        b_ref[i] = init_val;
        b_tst[i] = init_val;
      }

      // Unused entries of a partial batch must be zero.
      memset(buf, 0, sizeof(buf));
      memset(y, 0, sizeof(y));
      const int batch_size = 1 + (rnd_.Rand8() % WIENERNS_STATS_BATCH_SIZE);
      for (int p = 0; p < batch_size; ++p) {
        for (int k = 0; k < num_feat; ++k) {
          buf[k * WIENERNS_STATS_BATCH_SIZE + p] =
              static_cast<int16_t>((rnd_.Rand16() % 4096) - 2048);
        }
        y[p] = static_cast<int16_t>((rnd_.Rand16() % 4096) - 2048);
      }

      av2_accumulate_wienerns_correlation_batch_c(A_ref, b_ref, buf, y,
                                                  batch_size, num_feat);
      target_func_(A_tst, b_tst, buf, y, batch_size, num_feat);

      for (int k = 0; k < num_feat; ++k) {
        ASSERT_DOUBLE_EQ(b_ref[k], b_tst[k])
            << "b mismatch at index " << k << " for num_feat=" << num_feat;
      }

      for (int i = 0; i < num_feat; ++i) {
        for (int j = 0; j <= i; ++j) {
          ASSERT_DOUBLE_EQ(A_ref[i * num_feat + j], A_tst[i * num_feat + j])
              << "A mismatch at (" << i << ", " << j
              << ") for num_feat=" << num_feat;
        }
      }
    }
  }
}

void WienernsCorrTest::RunExtremeTest() {
  double A_ref[WIENERNS_TAPS_MAX * WIENERNS_TAPS_MAX];
  double A_tst[WIENERNS_TAPS_MAX * WIENERNS_TAPS_MAX];
  double b_ref[WIENERNS_TAPS_MAX];
  double b_tst[WIENERNS_TAPS_MAX];
  int16_t buf[WIENERNS_TAPS_MAX * WIENERNS_STATS_BATCH_SIZE];
  int16_t y[WIENERNS_STATS_BATCH_SIZE];

  const int16_t extreme_vals[] = { -32768, -4095, -2048, -1,   0,
                                   1,      2047,  4095,  32767 };
  const int num_extreme = sizeof(extreme_vals) / sizeof(extreme_vals[0]);

  for (int f_idx = 0; f_idx < kNumFeatTests; ++f_idx) {
    const int num_feat = kTestFeatCounts[f_idx];
    if (num_feat > WIENERNS_TAPS_MAX) continue;

    for (int e1 = 0; e1 < num_extreme; ++e1) {
      for (int e2 = 0; e2 < num_extreme; ++e2) {
        memset(A_ref, 0, sizeof(A_ref));
        memset(A_tst, 0, sizeof(A_tst));
        memset(b_ref, 0, sizeof(b_ref));
        memset(b_tst, 0, sizeof(b_tst));
        memset(buf, 0, sizeof(buf));
        memset(y, 0, sizeof(y));

        // A single pixel, so that the 32-bit partial sums cannot overflow.
        for (int k = 0; k < num_feat; ++k) {
          buf[k * WIENERNS_STATS_BATCH_SIZE] =
              (k % 2 == 0) ? extreme_vals[e1] : extreme_vals[e2];
        }
        y[0] = extreme_vals[e1];

        av2_accumulate_wienerns_correlation_batch_c(A_ref, b_ref, buf, y, 1,
                                                    num_feat);
        target_func_(A_tst, b_tst, buf, y, 1, num_feat);

        for (int k = 0; k < num_feat; ++k) {
          ASSERT_DOUBLE_EQ(b_ref[k], b_tst[k])
              << "Extreme b mismatch at index " << k
              << " for num_feat=" << num_feat;
        }

        for (int i = 0; i < num_feat; ++i) {
          for (int j = 0; j <= i; ++j) {
            ASSERT_DOUBLE_EQ(A_ref[i * num_feat + j], A_tst[i * num_feat + j])
                << "Extreme A mismatch at (" << i << ", " << j
                << ") for num_feat=" << num_feat;
          }
        }
      }
    }
  }
}

void WienernsCorrTest::RunSpeedTest(int run_times) {
  double A_ref[WIENERNS_TAPS_MAX * WIENERNS_TAPS_MAX] = { 0.0 };
  double A_tst[WIENERNS_TAPS_MAX * WIENERNS_TAPS_MAX] = { 0.0 };
  double b_ref[WIENERNS_TAPS_MAX] = { 0.0 };
  double b_tst[WIENERNS_TAPS_MAX] = { 0.0 };
  int16_t buf[WIENERNS_TAPS_MAX * WIENERNS_STATS_BATCH_SIZE];
  int16_t y[WIENERNS_STATS_BATCH_SIZE];
  const int batch_size = WIENERNS_STATS_BATCH_SIZE;

  for (int i = 0; i < WIENERNS_TAPS_MAX * WIENERNS_STATS_BATCH_SIZE; ++i) {
    buf[i] = static_cast<int16_t>((rnd_.Rand16() % 4096) - 2048);
  }
  for (int p = 0; p < batch_size; ++p) {
    y[p] = static_cast<int16_t>((rnd_.Rand16() % 4096) - 2048);
  }

  for (int f_idx = 0; f_idx < kNumFeatTests; ++f_idx) {
    const int num_feat = kTestFeatCounts[f_idx];
    if (num_feat > WIENERNS_TAPS_MAX) continue;

    avm_usec_timer timer;
    avm_usec_timer_start(&timer);
    for (int i = 0; i < run_times; ++i) {
      av2_accumulate_wienerns_correlation_batch_c(A_ref, b_ref, buf, y,
                                                  batch_size, num_feat);
    }
    avm_usec_timer_mark(&timer);
    const double time_c = static_cast<double>(avm_usec_timer_elapsed(&timer));

    avm_usec_timer_start(&timer);
    for (int i = 0; i < run_times; ++i) {
      target_func_(A_tst, b_tst, buf, y, batch_size, num_feat);
    }
    avm_usec_timer_mark(&timer);
    const double time_simd =
        static_cast<double>(avm_usec_timer_elapsed(&timer));

    printf("num_feat %2d: C %7.2fus / SIMD %7.2fus (speedup: %3.2fx)\n",
           num_feat, time_c, time_simd, time_c / time_simd);
  }
}

TEST_P(WienernsCorrTest, RandomValues) { RunRandomTest(); }
TEST_P(WienernsCorrTest, ExtremeValues) { RunExtremeTest(); }
TEST_P(WienernsCorrTest, DISABLED_Speed) { RunSpeedTest(20000); }

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(
    AVX2, WienernsCorrTest,
    ::testing::Values(av2_accumulate_wienerns_correlation_batch_avx2));
#endif  // HAVE_AVX2

typedef int64_t (*ComputeWienernsFeaturesFunc)(
    const uint16_t *dgd, const uint16_t *luma, const uint16_t *src, int width,
    const WienernsFeatureTaps *taps, int16_t *feat, int16_t *y);

class WienernsFeatureTest
    : public ::testing::TestWithParam<ComputeWienernsFeaturesFunc> {
 public:
  virtual void SetUp() {
    target_func_ = GetParam();
    rnd_.Reset(ACMRandom::DeterministicSeed());
  }
  virtual void TearDown() { libavm_test::ClearSystemState(); }

 protected:
  static const int kBorder = 8;
  static const int kStride = WIENERNS_STATS_BATCH_SIZE + 2 * kBorder + 3;
  static const int kRows = 2 * kBorder + 1;

  // Random taps within a +/-4 window; with_luma adds luma taps.
  void RandomTaps(WienernsFeatureTaps *taps, int with_luma) {
    memset(taps, 0, sizeof(*taps));
    taps->num_feat = 1 + rnd_.PseudoUniform(20);
    int nd = 0, nl = 0;
    for (int k = 0; k < taps->num_feat; ++k) {
      taps->feat_const[k] = (int16_t)rnd_.PseudoUniform(3);
      const int td = rnd_.PseudoUniform(4);
      for (int t = 0; t < td && nd < WIENERNS_FEAT_TAPS_MAX; ++t) {
        const int r = (int)rnd_.PseudoUniform(9) - 4;
        const int c = (int)rnd_.PseudoUniform(9) - 4;
        taps->dgd_offset[nd++] = r * kStride + c;
      }
      taps->dgd_start[k + 1] = nd;
      const int tl = with_luma ? rnd_.PseudoUniform(3) : 0;
      for (int t = 0; t < tl && nl < WIENERNS_FEAT_TAPS_MAX; ++t) {
        const int r = (int)rnd_.PseudoUniform(5) - 2;
        const int c = (int)rnd_.PseudoUniform(5) - 2;
        taps->luma_offset[nl++] = r * kStride + c;
      }
      taps->luma_start[k + 1] = nl;
    }
  }

  void RunCheck(int iters, int bd) {
    uint16_t dgd[kRows * kStride], luma[kRows * kStride], src[kRows * kStride];
    int16_t feat_ref[WIENERNS_TAPS_MAX * WIENERNS_STATS_BATCH_SIZE];
    int16_t feat_tst[WIENERNS_TAPS_MAX * WIENERNS_STATS_BATCH_SIZE];
    int16_t y_ref[WIENERNS_STATS_BATCH_SIZE], y_tst[WIENERNS_STATS_BATCH_SIZE];
    const int center = kBorder * kStride + kBorder;
    for (int iter = 0; iter < iters; ++iter) {
      const int with_luma = iter & 1;
      WienernsFeatureTaps taps;
      RandomTaps(&taps, with_luma);
      for (int i = 0; i < kRows * kStride; ++i) {
        dgd[i] = rnd_.Rand16() & ((1 << bd) - 1);
        luma[i] = rnd_.Rand16() & ((1 << bd) - 1);
        src[i] = rnd_.Rand16() & ((1 << bd) - 1);
      }
      const int width = 1 + rnd_.PseudoUniform(WIENERNS_STATS_BATCH_SIZE);
      memset(feat_ref, 0, sizeof(feat_ref));
      memset(feat_tst, 0, sizeof(feat_tst));
      const int64_t sse_ref = av2_compute_wienerns_features_c(
          dgd + center, with_luma ? luma + center : NULL, src + center, width,
          &taps, feat_ref, y_ref);
      const int64_t sse_tst =
          target_func_(dgd + center, with_luma ? luma + center : NULL,
                       src + center, width, &taps, feat_tst, y_tst);
      ASSERT_EQ(sse_ref, sse_tst) << "iter " << iter;
      for (int p = 0; p < width; ++p) {
        ASSERT_EQ(y_ref[p], y_tst[p]) << "iter " << iter << " p " << p;
        for (int k = 0; k < taps.num_feat; ++k) {
          ASSERT_EQ(feat_ref[k * WIENERNS_STATS_BATCH_SIZE + p],
                    feat_tst[k * WIENERNS_STATS_BATCH_SIZE + p])
              << "iter " << iter << " k " << k << " p " << p;
        }
      }
    }
  }

  void RunSpeed(int run_times) {
    uint16_t dgd[kRows * kStride], src[kRows * kStride];
    int16_t feat[WIENERNS_TAPS_MAX * WIENERNS_STATS_BATCH_SIZE];
    int16_t y[WIENERNS_STATS_BATCH_SIZE];
    for (int i = 0; i < kRows * kStride; ++i) {
      dgd[i] = rnd_.Rand16() & 1023;
      src[i] = rnd_.Rand16() & 1023;
    }
    WienernsFeatureTaps taps;
    memset(&taps, 0, sizeof(taps));
    // 16 features with 2 taps each, similar to the luma configuration.
    taps.num_feat = 17;
    for (int k = 0; k < 16; ++k) {
      taps.dgd_offset[2 * k] = (k % 4 + 1) * kStride + (k / 4) - 2;
      taps.dgd_offset[2 * k + 1] = -taps.dgd_offset[2 * k];
      taps.dgd_start[k + 1] = 2 * k + 2;
    }
    taps.dgd_start[17] = 32;
    taps.feat_const[16] = 1;
    const int center = kBorder * kStride + kBorder;
    avm_usec_timer timer;
    avm_usec_timer_start(&timer);
    for (int i = 0; i < run_times; ++i) {
      av2_compute_wienerns_features_c(dgd + center, NULL, src + center,
                                      WIENERNS_STATS_BATCH_SIZE, &taps, feat,
                                      y);
    }
    avm_usec_timer_mark(&timer);
    const double time_c = static_cast<double>(avm_usec_timer_elapsed(&timer));
    avm_usec_timer_start(&timer);
    for (int i = 0; i < run_times; ++i) {
      target_func_(dgd + center, NULL, src + center, WIENERNS_STATS_BATCH_SIZE,
                   &taps, feat, y);
    }
    avm_usec_timer_mark(&timer);
    const double time_simd =
        static_cast<double>(avm_usec_timer_elapsed(&timer));
    printf("features: C %7.2fus / SIMD %7.2fus (speedup: %3.2fx)\n", time_c,
           time_simd, time_c / time_simd);
  }

  ComputeWienernsFeaturesFunc target_func_;
  ACMRandom rnd_;
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(WienernsFeatureTest);

TEST_P(WienernsFeatureTest, RandomValues) {
  RunCheck(2000, 8);
  RunCheck(2000, 10);
  RunCheck(2000, 12);
}
TEST_P(WienernsFeatureTest, DISABLED_Speed) { RunSpeed(1000000); }

#if HAVE_AVX2
INSTANTIATE_TEST_SUITE_P(AVX2, WienernsFeatureTest,
                         ::testing::Values(av2_compute_wienerns_features_avx2));
#endif  // HAVE_AVX2

}  // namespace
