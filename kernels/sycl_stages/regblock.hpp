#pragma once
// Stage "regblock": S3, MR x NR register block per work-item

#include <sycl/sycl.hpp>

#include <cstddef>

#include "bench/config.hpp"
#include "bench/errors.hpp"
#include "gemm_args.hpp"

namespace kernels::regblock {

// Scratch memory this stage needs (bytes, multiple of sizeof(double))
inline std::size_t workspace_bytes(const bench::Problem&) { return 0; }

// Submits the work to q (in-order queue); the harness waits for completion
inline void run(sycl::queue& q, const GemmArgs& g) {
  (void)q;
  (void)g;
  throw bench::UnsupportedError("stage 'regblock' is not implemented");
}

} // namespace kernels::regblock
