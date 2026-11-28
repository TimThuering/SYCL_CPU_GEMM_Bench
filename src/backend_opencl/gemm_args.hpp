#pragma once
// Arguments of every OpenCL stage: the counterpart of GemmArgs (kernels/gemm_args.hpp) with
// sub-buffers instead of pointers. Same meaning of all fields; the kernels receive them in the
// order a, b, c, m, n, k, lda, ldb, ldc (+ workspace for stages that use one).

#include <cstddef>

#include "backend_opencl/arena.hpp"
#include "backend_opencl/cl_check.hpp"
#include "bench/config.hpp"

namespace bench {

struct ClGemmArgs {
  cl_mem a, b, c;
  cl_ulong m, n, k;             // ulong in the kernels (size_t is not allowed as a kernel argument)
  cl_ulong lda, ldb, ldc;
  cl_mem workspace;             // nullptr if workspace_bytes == 0
  std::size_t workspace_bytes;  // as requested by the stage, multiple of sizeof(double)
};

inline ClGemmArgs make_cl_gemm_args(const ClMatrices& mat, const Problem& p) {
  ClGemmArgs g;
  g.a = mat.a.get();
  g.b = mat.b.get();
  g.c = mat.c.get();
  g.m = p.m;
  g.n = p.n;
  g.k = p.k;
  g.lda = p.lda;
  g.ldb = p.ldb;
  g.ldc = p.ldc;
  g.workspace = mat.workspace.get();
  g.workspace_bytes = mat.workspace_bytes;
  return g;
}

} // namespace bench
