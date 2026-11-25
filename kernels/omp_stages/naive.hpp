#pragma once
// Stage "naive" (OpenMP twin): S1, one C element per loop iteration

#include <cstddef>

#include "bench/config.hpp"
#include "gemm_args.hpp"

namespace kernels::omp::naive {
    // Scratch memory this stage needs (bytes, multiple of sizeof(double))
    inline std::size_t workspace_bytes(const bench::Problem &) { return 0; }

    // Computes C; returns when done
    inline void run(const GemmArgs &g) {
        // Local copies, as in the SYCL version
        const double *a = g.a;
        const double *b = g.b;
        double *c = g.c;
        const std::size_t m = g.m, n = g.n, k = g.k;
        const std::size_t lda = g.lda, ldb = g.ldb, ldc = g.ldc;

        // One iteration per element of C (like the M x N range); collapse(2) distributes all m*n
        // (i, j) pairs over the threads, schedule(static) = fixed contiguous chunks per thread
#pragma omp parallel for collapse(2) schedule(static)
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (std::size_t kk = 0; kk < k; ++kk) {
                    sum += a[i * lda + kk] * b[kk * ldb + j];
                }
                c[i * ldc + j] = sum;
            }
        }
    }
} // namespace kernels::omp::naive
