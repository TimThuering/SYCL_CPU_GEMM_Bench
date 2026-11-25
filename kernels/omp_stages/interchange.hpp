#pragma once
// Stage "interchange" (OpenMP twin): S2, one C row per loop iteration, k outer / j inner

#include <cstddef>

#include "bench/config.hpp"
#include "gemm_args.hpp"

namespace kernels::omp::interchange {
    // Scratch memory this stage needs (bytes, multiple of sizeof(double))
    inline std::size_t workspace_bytes(const bench::Problem &) { return 0; }

    // Computes C; returns when done
    inline void run(const GemmArgs &g) {
        const double *a = g.a;
        const double *b = g.b;
        double *c = g.c;
        const std::size_t m = g.m, n = g.n, k = g.k;
        const std::size_t lda = g.lda, ldb = g.ldb, ldc = g.ldc;

#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; j++) {
                c[i * ldc + j] = 0.0;
            }

            for (std::size_t kk = 0; kk < k; kk++) {
                const double a_ik = a[i * lda + kk];
                for (std::size_t j = 0; j < n; j++) {
                    c[i * ldc + j] += a_ik * b[kk * ldb + j];
                }
            }
        }
    }
} // namespace kernels::omp::interchange
