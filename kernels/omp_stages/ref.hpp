#pragma once
// Stage "ref" (OpenMP twin): sequential triple loop. For testing the harness on small sizes only,
// not part of the study.

#include <cstddef>

#include "bench/config.hpp"
#include "gemm_args.hpp"

namespace kernels::omp::ref {

inline std::size_t workspace_bytes(const bench::Problem&) { return 0; }

inline void run(const GemmArgs& g) {
  for (std::size_t i = 0; i < g.m; ++i)
    for (std::size_t j = 0; j < g.n; ++j) {
      double sum = 0.0;
      for (std::size_t kk = 0; kk < g.k; ++kk) sum += g.a[i * g.lda + kk] * g.b[kk * g.ldb + j];
      g.c[i * g.ldc + j] = sum;
    }
}

} // namespace kernels::omp::ref
