#pragma once
// Stage "packed" (OpenCL reference), host part: S4, packed panels + KC/MC cache blocking around the
// S3 micro-kernel (kernels pack_b, pack_a, compute in packed.cl)

#include <algorithm>
#include <cstddef>

#include "backend_opencl/gemm_args.hpp"
#include "backend_opencl/program.hpp"
#include "bench/config.hpp"
#include "params.hpp"

namespace kernels::opencl::packed {
    // Block sizes (compile-time, kernels/params.hpp; the same values reach packed.cl as -D options)
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

    // KERNEL 1: pack rows pc … pc+kc-1 of B into Bp = ws[bp_offset ...]; one work-item per panel row
    inline void pack_b(bench::ClStageProgram &prog, cl_mem b, cl_ulong ldb, cl_ulong n, cl_ulong pc, cl_ulong kc,
                       cl_mem ws, cl_ulong bp_offset) {
        const cl_kernel kernel = prog.kernel("pack_b");
        bench::set_args(kernel, b, ldb, n, pc, ws, bp_offset);
        bench::enqueue_1d(prog.runtime(), kernel, kc);
    }

    // KERNEL 2: pack columns pc … pc+kc-1 of A into Ap = ws[0 ...]; one work-item per row incl. padding
    inline void pack_a(bench::ClStageProgram &prog, cl_mem a, cl_ulong lda, cl_ulong m, cl_ulong pc, cl_ulong kc,
                       cl_mem ws) {
        const std::size_t rows_p = ceil_div(m, MR) * MR;
        const cl_kernel kernel = prog.kernel("pack_a");
        bench::set_args(kernel, a, lda, m, pc, kc, ws);
        bench::enqueue_1d(prog.runtime(), kernel, rows_p);
    }

    // KERNEL 3: C (+)= Ap · Bp for the current panel; one work-item per (MC row block, NR column strip)
    inline void compute(bench::ClStageProgram &prog, cl_mem ws, cl_ulong bp_offset, cl_mem c, cl_ulong ldc,
                        cl_ulong m, cl_ulong n, cl_ulong kc, bool first) {
        const std::size_t blocks_m = ceil_div(m, MC); // row blocks of MC rows
        const std::size_t blocks_n = ceil_div(n, NR); // column strips of NR columns = Bp micro-panels
        const cl_kernel kernel = prog.kernel("compute");
        bench::set_args(kernel, ws, bp_offset, c, ldc, m, n, kc, cl_int{first ? 1 : 0});
        bench::enqueue_1d(prog.runtime(), kernel, blocks_m * blocks_n);
    }

    // Enqueues the work (in-order queue); the harness waits for completion
    inline void run(bench::ClStageProgram &prog, const bench::ClGemmArgs &g) {
        const std::size_t rows_p = ceil_div(g.m, MR) * MR;
        const cl_ulong bp_offset = rows_p * KC; // Bp follows Ap in the workspace (elements)

        // panel loop: three kernels per panel, in order (in-order queue)
        for (cl_ulong pc = 0; pc < g.k; pc += KC) {
            const cl_ulong kc = std::min<cl_ulong>(KC, g.k - pc); // last panel may be shorter
            const bool first = (pc == 0);
            pack_b(prog, g.b, g.ldb, g.n, pc, kc, g.workspace, bp_offset);         // KERNEL 1
            pack_a(prog, g.a, g.lda, g.m, pc, kc, g.workspace);                    // KERNEL 2
            compute(prog, g.workspace, bp_offset, g.c, g.ldc, g.m, g.n, kc, first); // KERNEL 3
        }
        // no clFinish(): the harness waits for completion
    }
} // namespace kernels::opencl::packed
