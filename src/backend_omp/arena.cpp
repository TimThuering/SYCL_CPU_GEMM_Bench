#include "arena.hpp"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "bench/datagen.hpp"
#include "bench/init_check.hpp"

namespace bench {

namespace {

constexpr std::size_t init_check_count = 32;  // random samples per matrix (plus corners)

// rows x ld matrix: value function for col < cols, NaN in the row padding
void fill_matrix(double* ptr, std::size_t rows, std::size_t cols, std::size_t ld, MatrixId id,
                 std::uint64_t seed) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  // Rows distributed over the OpenMP threads in contiguous blocks
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < ld; ++j)
      ptr[i * ld + j] = (j < cols) ? gen_value(seed, id, i, j) : nan;
}

// count consecutive doubles set to NaN
void fill_nan(double* ptr, std::size_t count) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < count; ++i) ptr[i] = nan;
}

} // namespace

Arena::Arena(const Problem& p, std::size_t workspace_bytes) : p_(p) {
  if (workspace_bytes % sizeof(double) != 0)
    throw std::invalid_argument("workspace size must be a multiple of sizeof(double)");

  // Offsets and sizes of A, B, C, workspace relative to the arena base
  const ArenaLayout layout = compute_layout(p, workspace_bytes);
  info_.alloc_kind = "std::aligned_alloc";
  info_.layout = layout;

  // Single page-aligned host allocation (total_bytes is a multiple of arena_page, as required)
  base_ = std::aligned_alloc(arena_page, layout.total_bytes);
  if (base_ == nullptr)
    throw std::runtime_error(info_.alloc_kind + " failed for " + std::to_string(layout.total_bytes) +
                             " bytes");
  try {
    check_arena_alignment(base_, info_.alloc_kind);
  } catch (...) {
    std::free(base_);  // the destructor does not run if the constructor throws
    throw;
  }

  // Region pointers: byte offsets on an unsigned char base, then typed as double
  auto* base = static_cast<unsigned char*>(base_);
  m_.a = reinterpret_cast<double*>(base + layout.a_offset);
  m_.b = reinterpret_cast<double*>(base + layout.b_offset);
  m_.c = reinterpret_cast<double*>(base + layout.c_offset);
  m_.workspace = workspace_bytes > 0 ? static_cast<void*>(base + layout.ws_offset) : nullptr;
  m_.workspace_bytes = workspace_bytes;
}

Arena::~Arena() { std::free(base_); }

void Arena::initialize(std::uint64_t seed) {
  // Parallel fill loops (each returns after all threads are done)
  fill_matrix(m_.a, p_.m, p_.k, p_.lda, MatrixId::A, seed);
  fill_matrix(m_.b, p_.k, p_.n, p_.ldb, MatrixId::B, seed);
  fill_nan(m_.c, p_.m * p_.ldc);
  if (m_.workspace != nullptr)
    fill_nan(static_cast<double*>(m_.workspace), m_.workspace_bytes / sizeof(double));

  // Read sampled elements (host memory: direct access) and compare with the host value function
  const auto samples = make_init_samples(p_, seed, init_check_count);
  std::vector<double> values(samples.size());
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const auto& s = samples[i];
    const double* src = s.matrix == InitSample::Matrix::A ? m_.a
                      : s.matrix == InitSample::Matrix::B ? m_.b
                                                          : m_.c;
    values[i] = src[s.offset];
  }
  check_init_samples(p_, seed, samples, values);
}

void Arena::copy_c_to_host(double* dst) const {
  // Whole C including the row padding (M x ldc elements)
  std::memcpy(dst, m_.c, info_.layout.c_bytes);
}

} // namespace bench
