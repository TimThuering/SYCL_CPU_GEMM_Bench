#pragma once
// Stage "interchange": S2, one C row per work-item, k outer / j inner

#include <sycl/sycl.hpp>

#include <cstddef>

#include "bench/config.hpp"
#include "gemm_args.hpp"

namespace kernels::interchange {
    // Scratch memory this stage needs (bytes, multiple of sizeof(double))
    inline std::size_t workspace_bytes(const bench::Problem &) { return 0; }

    // Submits the work to q (in-order queue); the harness waits for completion
    inline void run(sycl::queue &q, const GemmArgs &g) {
        const double *a = g.a;
        const double *b = g.b;
        double *c = g.c;
        const std::size_t m = g.m, n = g.n, k = g.k;
        const std::size_t lda = g.lda, ldb = g.ldb, ldc = g.ldc;

        // One work-item per row of C: range M; inside, k outer / j inner (contiguous in B and C)
        q.parallel_for(sycl::range<1>{m}, [=](sycl::id<1> idx) {
            const std::size_t i = idx[0];

            for (std::size_t j = 0; j < n; j++) {
                c[i * ldc + j] = 0.0;
            }

            for (std::size_t kk = 0; kk < k; kk++) {
                const double a_ik = a[i * lda + kk];
                for (std::size_t j = 0; j < n; j++) {
                    c[i * ldc + j] +=  a_ik * b[kk * ldb + j];
                }
            }

        });
    }
} // namespace kernels::interchange
