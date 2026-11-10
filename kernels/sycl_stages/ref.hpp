#pragma once
// Stage "ref": single_task triple loop. Sequential on one work-item; for testing the harness
// on small sizes only, not part of the study.

#include <sycl/sycl.hpp>

#include <cstddef>

#include "bench/config.hpp"
#include "gemm_args.hpp"

namespace kernels::ref {

inline std::size_t workspace_bytes(const bench::Problem&) { return 0; }

inline void run(sycl::queue& q, const GemmArgs& g) {
  q.single_task([=]() {
    for (std::size_t i = 0; i < g.m; ++i)
      for (std::size_t j = 0; j < g.n; ++j) {
        double sum = 0.0;
        for (std::size_t kk = 0; kk < g.k; ++kk) sum += g.a[i * g.lda + kk] * g.b[kk * g.ldb + j];
        g.c[i * g.ldc + j] = sum;
      }
  });
}

} // namespace kernels::ref
