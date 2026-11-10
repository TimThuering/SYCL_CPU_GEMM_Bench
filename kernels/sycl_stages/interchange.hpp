#pragma once
// Stage "interchange": S2, one C row per work-item, k outer / j inner

#include <sycl/sycl.hpp>

#include <cstddef>

#include "bench/config.hpp"
#include "bench/errors.hpp"
#include "gemm_args.hpp"

namespace kernels::interchange {

// Scratch memory this stage needs (bytes, multiple of sizeof(double))
inline std::size_t workspace_bytes(const bench::Problem&) { return 0; }

// Submits the work to q (in-order queue); the harness waits for completion
inline void run(sycl::queue& q, const GemmArgs& g) {
  (void)q;
  (void)g;
  throw bench::UnsupportedError("stage 'interchange' is not implemented");
}

} // namespace kernels::interchange
