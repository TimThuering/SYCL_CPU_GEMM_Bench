#pragma once
// Stage "packed": S4, packed panels + KC/MC cache blocking around the S3 micro-kernel

#include <sycl/sycl.hpp>

#include <cstddef>

#include "bench/config.hpp"
#include "bench/errors.hpp"
#include "gemm_args.hpp"

namespace kernels::packed {

// Scratch memory this stage needs (bytes, multiple of sizeof(double))
inline std::size_t workspace_bytes(const bench::Problem&) { return 0; }

// Submits the work to q (in-order queue); the harness waits for completion
inline void run(sycl::queue& q, const GemmArgs& g) {
  (void)q;
  (void)g;
  throw bench::UnsupportedError("stage 'packed' is not implemented");
}

} // namespace kernels::packed
