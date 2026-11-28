#pragma once
// Stage "naive" (OpenCL reference), host part: S1, one C element per work-item (kernel in naive.cl)

#include <cstddef>

#include "backend_opencl/gemm_args.hpp"
#include "backend_opencl/program.hpp"
#include "bench/config.hpp"

namespace kernels::opencl::naive {
    // Scratch memory this stage needs (bytes, multiple of sizeof(double))
    inline std::size_t workspace_bytes(const bench::Problem &) { return 0; }

    // Enqueues the work (in-order queue); the harness waits for completion
    inline void run(bench::ClStageProgram &prog, const bench::ClGemmArgs &g) {
        const cl_kernel kernel = prog.kernel("gemm_naive");
        bench::set_args(kernel, g.a, g.b, g.c, g.m, g.n, g.k, g.lda, g.ldb, g.ldc);

        // SYCL range<2>{m, n} (j fastest) = OpenCL global {n, m} (dimension 0 fastest)
        bench::enqueue_2d(prog.runtime(), kernel, g.n, g.m);
    }
} // namespace kernels::opencl::naive
