#include "bench/layout.hpp"

#include <cstdint>
#include <ostream>
#include <stdexcept>

namespace bench {

namespace {

// Smallest multiple of `align` that is >= x, e.g. round_up(5000, 4096) = 8192
std::size_t round_up(std::size_t x, std::size_t align) {
  return (x + align - 1) / align * align;
}

} // namespace

// Sizes are bounded by the CLI (every dimension <= 1e6), so no overflow checks are needed here.
ArenaLayout compute_layout(const Problem& p, std::size_t workspace_bytes) {
  ArenaLayout l;
  l.a_bytes = p.m * p.lda * sizeof(double);
  l.b_bytes = p.k * p.ldb * sizeof(double);
  l.c_bytes = p.m * p.ldc * sizeof(double);
  l.ws_bytes = workspace_bytes;

  // Each region starts at the next page boundary after the previous one
  l.a_offset = 0;
  l.b_offset = round_up(l.a_offset + l.a_bytes, arena_page);
  l.c_offset = round_up(l.b_offset + l.b_bytes, arena_page);
  l.ws_offset = round_up(l.c_offset + l.c_bytes, arena_page);
  // Whole pages (std::aligned_alloc requires a multiple of the alignment)
  l.total_bytes = round_up(l.ws_offset + l.ws_bytes, arena_page);
  return l;
}

std::size_t page_offset(const void* ptr) {
  return static_cast<std::size_t>(reinterpret_cast<std::uintptr_t>(ptr) % arena_page);
}

void record_alignment(ArenaInfo& info, const void* base, bool require) {
  info.requested_alignment = arena_page;
  info.base_mod_page = page_offset(base);
  info.alignment_honored = (info.base_mod_page == 0);
  if (require && !info.alignment_honored)
    throw std::runtime_error(info.alloc_kind + " returned an address that is not " +
                             std::to_string(arena_page) + "-byte aligned (address mod " +
                             std::to_string(arena_page) + " = " + std::to_string(info.base_mod_page) +
                             ") and --require-alignment is set");
}

void print_arena_info(std::ostream& os, const ArenaInfo& info) {
  const auto& l = info.layout;
  os << "arena:         " << info.alloc_kind << ", " << l.total_bytes << " bytes, "
     << info.requested_alignment << "-byte alignment requested, base mod " << arena_page << " = "
     << info.base_mod_page << (info.alignment_honored ? " (honored)" : " (NOT honored)") << "\n"
     << "  A  offset " << l.a_offset << ", " << l.a_bytes << " bytes\n"
     << "  B  offset " << l.b_offset << ", " << l.b_bytes << " bytes\n"
     << "  C  offset " << l.c_offset << ", " << l.c_bytes << " bytes\n"
     << "  ws offset " << l.ws_offset << ", " << l.ws_bytes << " bytes\n";
}

} // namespace bench
