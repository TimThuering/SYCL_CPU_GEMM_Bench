#pragma once
// Stage "interchange" (OpenMP twin): S2, one C row per loop iteration, k outer / j inner

#include <cstddef>

#include "bench/config.hpp"
#include "bench/errors.hpp"
#include "gemm_args.hpp"

namespace kernels::omp::interchange {

// Scratch memory this stage needs (bytes, multiple of sizeof(double))
inline std::size_t workspace_bytes(const bench::Problem&) { return 0; }

// Computes C; returns when done
inline void run(const GemmArgs& g) {
  (void)g;
  throw bench::UnsupportedError("stage 'interchange' is not implemented");
}

} // namespace kernels::omp::interchange
