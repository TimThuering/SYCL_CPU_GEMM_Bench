#pragma once
// Stage "regblock" (OpenMP twin): S3, MR x NR register block per loop iteration

#include <cstddef>

#include "bench/config.hpp"
#include "bench/errors.hpp"
#include "gemm_args.hpp"

namespace kernels::omp::regblock {

// Scratch memory this stage needs (bytes, multiple of sizeof(double))
inline std::size_t workspace_bytes(const bench::Problem&) { return 0; }

// Computes C; returns when done
inline void run(const GemmArgs& g) {
  (void)g;
  throw bench::UnsupportedError("stage 'regblock' is not implemented");
}

} // namespace kernels::omp::regblock
