#pragma once
// Stage "packed" (OpenMP twin): S4, packed panels + KC/MC cache blocking around the S3 micro-kernel
// Same algorithm, layout, loop structure and expressions as kernels/sycl_stages/packed.hpp; each SYCL
// kernel is an OpenMP parallel loop over the same iteration space (range dimension 0 = outer loop,
// dimension 1 = inner loop). The implicit barrier at the end of each loop gives the same order as the
// in-order queue.

#include <algorithm>
#include <cstddef>

#include "bench/config.hpp"
#include "gemm_args.hpp"
#include "params.hpp"

namespace kernels::omp::packed {
    // Block sizes (compile-time, kernels/params.hpp):
    //   MR x NR  micro-tile computed in registers
    //   KC       panel length along K (columns of A / rows of B packed at a time)
    //   MC       rows of C per iteration in the compute loop (multiple of MR)
    inline constexpr std::size_t MR = gemm::MR;
    inline constexpr std::size_t NR = gemm::NR;
    inline constexpr std::size_t KC = gemm::KC;
    inline constexpr std::size_t MC = gemm::MC;

    // ceil(a / b) for positive integers, e.g. number of micro-panels needed to cover a rows
    inline constexpr std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

    // Scratch memory this stage needs (bytes, multiple of sizeof(double))
    // Ap: rows_p x KC (M rounded up to whole MR micro-panels), Bp: KC x cols_p (N rounded up to NR)
    inline std::size_t workspace_bytes(const bench::Problem &p) {
        const std::size_t rows_p = ceil_div(p.m, MR) * MR;
        const std::size_t cols_p = ceil_div(p.n, NR) * NR;
        return (rows_p + cols_p) * KC * sizeof(double);
    }

    // KERNEL 1: pack rows pc … pc+kc-1 of B (all n columns) into NR-wide micro-panels
    //   bp[(jr·KC + p)·NR + s] = B(pc + p, jr·NR + s),  0 for columns >= n (zero padding)
    inline void pack_b(const double *b, std::size_t ldb, std::size_t n,
                       std::size_t pc, std::size_t kc, double *bp) {
        const std::size_t cols_p = ceil_div(n, NR) * NR;

        // one iteration per panel row p: copies row pc + p of B sequentially (contiguous reads),
        // NR values at a time into micro-panel jr (SYCL: range<1>{kc})
#pragma omp parallel for schedule(static)
        for (std::size_t p = 0; p < kc; ++p) {                // k step within the panel   (0 … kc-1)
            for (std::size_t jj = 0; jj < cols_p; ++jj) {     // column incl. padding      (0 … cols_p-1)
                const std::size_t jr = jj / NR; // which micro-panel
                const std::size_t s = jj % NR;  // position inside it
                const double value = (jj < n) ? b[(pc + p) * ldb + jj] : 0.0; // global row = pc + p
                bp[(jr * KC + p) * NR + s] = value;
            }
        }
    }

    // KERNEL 2: pack columns pc … pc+kc-1 of A (all m rows) into MR-tall micro-panels
    //   ap[(ir·KC + p)·MR + r] = A(ir·MR + r, pc + p),  0 for rows >= m (zero padding)
    inline void pack_a(const double *a, std::size_t lda, std::size_t m,
                       std::size_t pc, std::size_t kc, double *ap) {
        const std::size_t rows_p = ceil_div(m, MR) * MR;

        // one iteration per row ii (incl. padding rows): copies the kc panel values of row ii of A
        // sequentially (contiguous reads) into micro-panel ir, at position r of every k step
        // (SYCL: range<1>{rows_p})
#pragma omp parallel for schedule(static)
        for (std::size_t ii = 0; ii < rows_p; ++ii) {   // row incl. padding         (0 … rows_p-1)
            const std::size_t ir = ii / MR; // which micro-panel
            const std::size_t r = ii % MR;  // position inside it
            for (std::size_t p = 0; p < kc; ++p) {      // k step within the panel   (0 … kc-1)
                const double value = (ii < m) ? a[ii * lda + (pc + p)] : 0.0; // global column = pc + p
                ap[(ir * KC + p) * MR + r] = value;
            }
        }
    }

    // KERNEL 3: C (+)= Ap · Bp for the current panel
    //   iteration (ic, jr): MC rows x NR columns of C = up to MC/MR micro-tiles, all with Bp micro-panel jr
    //   first panel (first = true): C = result, later panels: C += result
    inline void compute(const double *ap, const double *bp, double *c, std::size_t ldc,
                        std::size_t m, std::size_t n, std::size_t kc, bool first) {
        const std::size_t blocks_m = ceil_div(m, MC); // row blocks of MC rows
        const std::size_t blocks_n = ceil_div(n, NR); // column strips of NR columns = Bp micro-panels
        const std::size_t tiles_m = ceil_div(m, MR);  // Ap micro-panels in total

        // 1D, t = ic · blocks_n + jr (SYCL: range<1>{blocks_m * blocks_n}): consecutive iterations share ic
        // (same Ap micro-panels), so a contiguous chunk of t per thread reuses its Ap block from cache
#pragma omp parallel for schedule(static)
        for (std::size_t t = 0; t < blocks_m * blocks_n; ++t) {
            const std::size_t ic = t / blocks_n; // row block of MC rows
            const std::size_t jr = t % blocks_n; // column strip of NR columns
            const std::size_t ir_begin = ic * (MC / MR);
            const std::size_t ir_end = (ir_begin + MC / MR < tiles_m) ? ir_begin + MC / MR : tiles_m;

            const double *bpanel = bp + jr * KC * NR; // Bp micro-panel jr (reused for all micro-tiles)

            for (std::size_t ir = ir_begin; ir < ir_end; ++ir) {
                const double *apanel = ap + ir * KC * MR; // Ap micro-panel ir

                // micro-kernel: always a full MR x NR tile (padding in Ap/Bp is zero)
                double acc[MR][NR] = {};
                for (std::size_t p = 0; p < kc; ++p) {
                    double a_r[MR];
                    double b_s[NR];
                    for (std::size_t r = 0; r < MR; ++r) a_r[r] = apanel[p * MR + r];
                    for (std::size_t s = 0; s < NR; ++s) b_s[s] = bpanel[p * NR + s];
                    for (std::size_t r = 0; r < MR; ++r)
                        for (std::size_t s = 0; s < NR; ++s)
                            acc[r][s] += a_r[r] * b_s[s];
                }

                // write-back, masked at the matrix edge (padding results are dropped)
                for (std::size_t r = 0; r < MR; ++r) {
                    const std::size_t i = ir * MR + r;
                    if (i >= m) break;
                    for (std::size_t s = 0; s < NR; ++s) {
                        const std::size_t j = jr * NR + s;
                        if (j >= n) break;
                        if (first) c[i * ldc + j] = acc[r][s];
                        else       c[i * ldc + j] += acc[r][s];
                    }
                }
            }
        }
    }

    // Computes C; returns when done
    inline void run(const GemmArgs &g) {
        // local copies + workspace split
        const double *a = g.a, *b = g.b;
        double *c = g.c;                            // output: written by the compute loop
        const std::size_t m = g.m, n = g.n, k = g.k;
        const std::size_t lda = g.lda, ldb = g.ldb, ldc = g.ldc;
        const std::size_t rows_p = ceil_div(m, MR) * MR;
        double *ap = static_cast<double *>(g.workspace); // rows_p · KC doubles
        double *bp = ap + rows_p * KC;                   // cols_p · KC doubles

        // panel loop: three parallel loops per panel, in order (implicit barrier after each)
        for (std::size_t pc = 0; pc < k; pc += KC) {
            const std::size_t kc = std::min(KC, k - pc); // last panel may be shorter
            const bool first = (pc == 0);
            pack_b(b, ldb, n, pc, kc, bp);               // KERNEL 1
            pack_a(a, lda, m, pc, kc, ap);               // KERNEL 2
            compute(ap, bp, c, ldc, m, n, kc, first);    // KERNEL 3
        }
    }
} // namespace kernels::omp::packed
