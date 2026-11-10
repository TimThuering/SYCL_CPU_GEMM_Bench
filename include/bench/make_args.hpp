#pragma once
// Kernel arguments from the arena pointers and the problem.

#include "bench/config.hpp"
#include "bench/layout.hpp"
#include "gemm_args.hpp"

namespace bench {

inline GemmArgs make_gemm_args(const Matrices& mat, const Problem& p) {
  GemmArgs g;
  g.a = mat.a;
  g.b = mat.b;
  g.c = mat.c;
  g.m = p.m;
  g.n = p.n;
  g.k = p.k;
  g.lda = p.lda;
  g.ldb = p.ldb;
  g.ldc = p.ldc;
  g.workspace = mat.workspace;
  g.workspace_bytes = mat.workspace_bytes;
  return g;
}

} // namespace bench
