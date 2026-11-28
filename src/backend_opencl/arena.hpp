#pragma once
// Arena for the OpenCL reference: one buffer allocated by the OpenCL implementation
// (clCreateBuffer, no host pointer), with one sub-buffer per region (A, B, C, workspace) at the
// page offsets of the common layout (bench/layout.hpp).
//
// clCreateBuffer has no alignment argument, so the alignment of the base cannot be requested; the
// base address as seen by the kernels is recorded (base_mod_page, alignment_honored = page-aligned).
// With --require-alignment, a base that is not page-aligned aborts the run.

#include <cstddef>
#include <cstdint>

#include "bench/config.hpp"
#include "bench/layout.hpp"
#include "backend_opencl/handle.hpp"
#include "backend_opencl/runtime.hpp"

namespace bench {

// Sub-buffers of the arena (workspace: empty handle if the stage needs none)
struct ClMatrices {
  ClMem a, b, c, workspace;
  std::size_t workspace_bytes = 0;
};

class Arena {
 public:
  Arena(const ClRuntime& rt, const Problem& p, std::size_t workspace_bytes, bool require_alignment = false);
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  // Parallel init kernels (first touch by the worker threads):
  //   A, B: gen_value(seed, ...), row padding = NaN;  C, workspace: NaN sentinel.
  // Waits for completion, then runs the init sample check (throws on mismatch).
  void initialize(std::uint64_t seed);

  // Copies C (M x ldc elements) into host memory dst. Blocking.
  void copy_c_to_host(double* dst) const;

  const ClMatrices& matrices() const { return m_; }
  const ArenaInfo& info() const { return info_; }

 private:
  const ClRuntime& rt_;
  Problem p_;
  ClProgram init_program_;
  ClMem base_;     // declared before the sub-buffers: they are released first
  ClMatrices m_;
  ArenaInfo info_;
};

} // namespace bench
