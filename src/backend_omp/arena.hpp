#pragma once
// Arena for the OpenMP twin: one page-aligned host allocation, same layout as the SYCL arena.

#include <cstddef>
#include <cstdint>

#include "bench/config.hpp"
#include "bench/layout.hpp"

namespace bench {

class Arena {
 public:
  Arena(const Problem& p, std::size_t workspace_bytes);
  ~Arena();
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  // OpenMP parallel init (first touch by the worker threads), same values as the SYCL arena:
  //   A, B: gen_value(seed, ...), row padding = NaN;  C, workspace: NaN sentinel.
  // Then runs the init sample check (throws on mismatch).
  void initialize(std::uint64_t seed);

  // Copies C (M x ldc elements) into dst.
  void copy_c_to_host(double* dst) const;

  const Matrices& matrices() const { return m_; }
  const ArenaInfo& info() const { return info_; }

 private:
  Problem p_;
  void* base_ = nullptr;
  Matrices m_;
  ArenaInfo info_;
};

} // namespace bench
