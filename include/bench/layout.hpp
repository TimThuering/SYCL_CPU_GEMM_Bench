#pragma once
// Memory layout of the arena: one allocation holding A, B, C and the stage workspace.
//
//   [ A: M x lda ][pad][ B: K x ldb ][pad][ C: M x ldc ][pad][ workspace ]
//
// The arena base is allocated page-aligned (checked after allocation) and every region starts
// at a page boundary, so the layout is identical for all implementations.
//
// Workspace: scratch memory a stage needs in addition to A, B, C (e.g. the packing buffers
// of the packed stage); 0 bytes for stages without one. It lives in the arena so it is
// allocated and first-touched once during setup, never inside the timed iterations.

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>

#include "bench/config.hpp"

namespace bench {

inline constexpr std::size_t arena_page = 4096;  // base and region alignment in bytes

struct ArenaLayout {
  std::size_t a_offset = 0, a_bytes = 0;
  std::size_t b_offset = 0, b_bytes = 0;
  std::size_t c_offset = 0, c_bytes = 0;
  std::size_t ws_offset = 0, ws_bytes = 0;
  std::size_t total_bytes = 0;  // allocation size, rounded up to a multiple of arena_page
};

ArenaLayout compute_layout(const Problem& p, std::size_t workspace_bytes);

// Throws std::runtime_error if ptr is not a multiple of arena_page
void check_arena_alignment(const void* ptr, const std::string& alloc_kind);

// Recorded with every result
struct ArenaInfo {
  std::string alloc_kind;           // e.g. "sycl::aligned_alloc_device", "std::aligned_alloc"
  ArenaLayout layout;
  std::size_t alignment = arena_page;
};

void print_arena_info(std::ostream& os, const ArenaInfo& info);

// Raw pointers into the arena (device pointers for SYCL)
struct Matrices {
  double* a = nullptr;
  double* b = nullptr;
  double* c = nullptr;
  void* workspace = nullptr;
  std::size_t workspace_bytes = 0;
};

} // namespace bench
