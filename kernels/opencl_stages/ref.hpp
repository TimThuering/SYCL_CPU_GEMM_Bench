#pragma once
// Stage "ref" (OpenCL reference), host part: one work-item runs the kernel gemm_ref (ref.cl).

#include <cstddef>

#include "backend_opencl/gemm_args.hpp"
#include "backend_opencl/program.hpp"
#include "bench/config.hpp"

namespace kernels::opencl::ref {

inline std::size_t workspace_bytes(const bench::Problem&) { return 0; }

inline void run(bench::ClStageProgram& prog, const bench::ClGemmArgs& g) {
  const cl_kernel kernel = prog.kernel("gemm_ref");
  bench::set_args(kernel, g.a, g.b, g.c, g.m, g.n, g.k, g.lda, g.ldb, g.ldc);
  bench::enqueue_1d(prog.runtime(), kernel, 1);
}

} // namespace kernels::opencl::ref
