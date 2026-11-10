#pragma once
// Stage "packed" (OpenMP twin): S4, packed panels + KC/MC cache blocking around the S3 micro-kernel

#include <cstddef>

#include "bench/config.hpp"
#include "bench/errors.hpp"
#include "gemm_args.hpp"

namespace kernels::omp::packed {

// Scratch memory this stage needs (bytes, multiple of sizeof(double))
inline std::size_t workspace_bytes(const bench::Problem&) { return 0; }

// Computes C; returns when done
inline void run(const GemmArgs& g) {
  (void)g;
  throw bench::UnsupportedError("stage 'packed' is not implemented");
}

} // namespace kernels::omp::packed
