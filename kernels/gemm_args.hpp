#pragma once
// Arguments of every GEMM stage (SYCL and OpenMP twin).
//
// Computes C = A * B (double, row-major):
//   A: m x k, element (i, kk) at a[i * lda + kk]
//   B: k x n, element (kk, j) at b[kk * ldb + j]
//   C: m x n, element (i, j)  at c[i * ldc + j]
// A stage must overwrite C (not accumulate onto its previous content): C holds a NaN sentinel
// before the first iteration. Columns n .. ldc-1 (row padding) must not be written.
//
// Kernels get everything they need through this struct, captured by value in the kernel lambda.
// No access to global variables (e.g. the harness configuration) inside kernels.

#include <cstddef>

struct GemmArgs {
  const double* a;
  const double* b;
  double* c;
  std::size_t m, n, k;
  std::size_t lda, ldb, ldc;
  void* workspace;              // stage scratch memory (nullptr if workspace_bytes == 0)
  std::size_t workspace_bytes;  // as requested by the stage, multiple of sizeof(double)
};
