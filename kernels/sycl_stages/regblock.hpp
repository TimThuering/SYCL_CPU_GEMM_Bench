#pragma once
// Stage "regblock": S3, MR x NR register block per work-item

#include <sycl/sycl.hpp>

#include <cstddef>

#include "bench/config.hpp"
#include "gemm_args.hpp"
#include "params.hpp"

namespace kernels::regblock {
    // Scratch memory this stage needs (bytes, multiple of sizeof(double))
    inline std::size_t workspace_bytes(const bench::Problem &) { return 0; }

    // Submits the work to q (in-order queue); the harness waits for completion
    inline void run(sycl::queue &q, const GemmArgs &g) {
        const double *a = g.a;
        const double *b = g.b;
        double *c = g.c;
        const std::size_t m = g.m, n = g.n, k = g.k;
        const std::size_t lda = g.lda, ldb = g.ldb, ldc = g.ldc;

        const std::size_t blocks_m = (m + gemm::MR - 1) / gemm::MR; // ceil(m / MR)
        const std::size_t blocks_n = (n + gemm::NR - 1) / gemm::NR;
        q.parallel_for(sycl::range<2>{blocks_m, blocks_n}, [=](sycl::id<2> idx) {
            const std::size_t i = idx[0];
            const std::size_t j = idx[1];

            // small arrays to be kept in registers
            double acc_r[gemm::MR][gemm::NR] = {};
            double a_r[gemm::MR] = {};
            double b_r[gemm::NR] = {};

            const std::size_t i0 = i * gemm::MR;
            const std::size_t j0 = j * gemm::NR;
            const bool full = (i0 + gemm::MR <= m) && (j0 + gemm::NR <= n);

            if (full) {
                // no need for border checks
                for (std::size_t kk = 0; kk < k; ++kk) {
                    // load a_r values
                    for (std::size_t ii = 0; ii < gemm::MR; ++ii) {
                            a_r[ii] = a[(i0 + ii) * lda + kk];
                    }

                    // load b_r values
                    for (std::size_t jj = 0; jj < gemm::NR; ++jj) {
                            b_r[jj] = b[kk * ldb + j0 + jj];
                    }

                    // a_r^T * b_r (outer product)
                    for (std::size_t ii = 0; ii < gemm::MR; ++ii) {
                        for (std::size_t jj = 0; jj < gemm::NR; ++jj) {
                            acc_r[ii][jj] += a_r[ii] * b_r[jj];
                        }
                    }
                }

                // write back local result to c
                for (std::size_t ii = 0; ii < gemm::MR; ++ii) {
                    for (std::size_t jj = 0; jj < gemm::NR; ++jj) {
                            c[(i0 + ii) * ldc + j0 + jj] = acc_r[ii][jj];
                    }
                }
            } else {
                // check borders as block does not fully overlay with matrix
                for (std::size_t kk = 0; kk < k; ++kk) {
                    // load a_r values
                    for (std::size_t ii = 0; ii < gemm::MR; ++ii) {
                        if (i0 + ii < m) {
                            a_r[ii] = a[(i0 + ii) * lda + kk];
                        }
                    }

                    // load b_r values
                    for (std::size_t jj = 0; jj < gemm::NR; ++jj) {
                        if (j0 + jj < n) {
                            b_r[jj] = b[kk * ldb + j0 + jj];
                        }
                    }

                    // a_r^T * b_r (outer product)
                    for (std::size_t ii = 0; ii < gemm::MR; ++ii) {
                        for (std::size_t jj = 0; jj < gemm::NR; ++jj) {
                            acc_r[ii][jj] += a_r[ii] * b_r[jj];
                        }
                    }
                }

                // write back local result to c
                for (std::size_t ii = 0; ii < gemm::MR; ++ii) {
                    for (std::size_t jj = 0; jj < gemm::NR; ++jj) {
                        if (i0 + ii < m && j0 + jj < n) {
                            c[(i0 + ii) * ldc + j0 + jj] = acc_r[ii][jj];
                        }
                    }
                }
            }
        });
    }
} // namespace kernels::regblock
