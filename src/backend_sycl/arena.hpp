#pragma once
// Arena for the SYCL driver: one sycl::aligned_alloc_device allocation (arena_page-aligned, checked).

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

#include "bench/config.hpp"
#include "bench/layout.hpp"

namespace bench {

class Arena {
 public:
  Arena(sycl::queue& q, const Problem& p, std::size_t workspace_bytes);
  ~Arena();
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  // Parallel init kernels (first touch by the worker threads):
  //   A, B: gen_value(seed, ...), row padding = NaN;  C, workspace: NaN sentinel.
  // Waits for completion, then runs the init sample check (throws on mismatch).
  void initialize(std::uint64_t seed);

  // Copies C (M x ldc elements) into host memory dst. Blocking.
  void copy_c_to_host(double* dst) const;

  const Matrices& matrices() const { return m_; }
  const ArenaInfo& info() const { return info_; }

 private:
  sycl::queue& q_;
  Problem p_;
  void* base_ = nullptr;
  Matrices m_;
  ArenaInfo info_;
};

} // namespace bench
