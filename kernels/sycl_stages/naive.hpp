#pragma once
// Stage "naive": S1, one C element per work-item

#include <sycl/sycl.hpp>

#include <cstddef>

#include "bench/config.hpp"
#include "gemm_args.hpp"

namespace kernels::naive {
    // Scratch memory this stage needs (bytes, multiple of sizeof(double))
    inline std::size_t workspace_bytes(const bench::Problem &) { return 0; }

    // Submits the work to q (in-order queue); the harness waits for completion
    inline void run(sycl::queue &q, const GemmArgs &g) {
        const double *a = g.a;
        const double *b = g.b;
        double *c = g.c;
        const std::size_t m = g.m, n = g.n, k = g.k;
        const std::size_t lda = g.lda, ldb = g.ldb, ldc = g.ldc;

        // One work-item per element of C: range M x N, last dimension (idx[1]) = fastest = j
        q.parallel_for(sycl::range<2>{m, n}, [=](sycl::id<2> idx) {
            const std::size_t i = idx[0];
            const std::size_t j = idx[1];

            double sum = 0.0;
            for (std::size_t kk = 0; kk < k; ++kk) {
                sum += a[i * lda + kk] * b[kk * ldb + j];
            }
            c[i * ldc + j] = sum;
        });
    }
} // namespace kernels::naive
