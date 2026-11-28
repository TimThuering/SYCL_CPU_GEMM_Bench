#pragma once
// Stage "interchange" (OpenCL reference), host part: S2, one C row per work-item (kernel in interchange.cl)

#include <cstddef>

#include "backend_opencl/gemm_args.hpp"
#include "backend_opencl/program.hpp"
#include "bench/config.hpp"

namespace kernels::opencl::interchange {
    // Scratch memory this stage needs (bytes, multiple of sizeof(double))
    inline std::size_t workspace_bytes(const bench::Problem &) { return 0; }

    // Enqueues the work (in-order queue); the harness waits for completion
    inline void run(bench::ClStageProgram &prog, const bench::ClGemmArgs &g) {
        const cl_kernel kernel = prog.kernel("gemm_interchange");
        bench::set_args(kernel, g.a, g.b, g.c, g.m, g.n, g.k, g.lda, g.ldb, g.ldc);

        // SYCL range<1>{m} = OpenCL global {m}
        bench::enqueue_1d(prog.runtime(), kernel, g.m);
    }
} // namespace kernels::opencl::interchange
