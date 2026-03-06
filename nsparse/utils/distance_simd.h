#ifndef DISTANCE_SIMD_H
#define DISTANCE_SIMD_H

#include "nsparse/sparse_vectors.h"

#if defined(__AVX512F__)
#include "nsparse/utils/distance_avx512.h"
#elif defined(__AVX2__)
#include "nsparse/utils/distance_avx2.h"
#elif defined(__ARM_FEATURE_SVE)
#include "nsparse/utils/distance_sve.h"
#elif defined(__aarch64__)
#include "nsparse/utils/distance_neon.h"
#else
#include "nsparse/utils/distance.h"
#endif

#endif
