#pragma once
// Stage "regblock" (OpenCL reference), host part: S3, MR x NR register block per work-item
// (kernel in regblock.cl)

#include <cstddef>

#include "backend_opencl/gemm_args.hpp"
#include "backend_opencl/program.hpp"
#include "bench/config.hpp"
#include "params.hpp"

namespace kernels::opencl::regblock {
    // Scratch memory this stage needs (bytes, multiple of sizeof(double))
    inline std::size_t workspace_bytes(const bench::Problem &) { return 0; }

    // Enqueues the work (in-order queue); the harness waits for completion
    inline void run(bench::ClStageProgram &prog, const bench::ClGemmArgs &g) {
        const cl_kernel kernel = prog.kernel("gemm_regblock");
        bench::set_args(kernel, g.a, g.b, g.c, g.m, g.n, g.k, g.lda, g.ldb, g.ldc);

        const std::size_t blocks_m = (g.m + gemm::MR - 1) / gemm::MR; // ceil(m / MR)
        const std::size_t blocks_n = (g.n + gemm::NR - 1) / gemm::NR;
        // SYCL range<2>{blocks_m, blocks_n} = OpenCL global {blocks_n, blocks_m}
        bench::enqueue_2d(prog.runtime(), kernel, blocks_n, blocks_m);
    }
} // namespace kernels::opencl::regblock
