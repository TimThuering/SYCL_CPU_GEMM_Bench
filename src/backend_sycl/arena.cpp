#include "arena.hpp"

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
void fill_matrix(sycl::queue& q, double* ptr, std::size_t rows, std::size_t cols, std::size_t ld,
                 MatrixId id, std::uint64_t seed) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  // One work-item per element, including the padding columns (j >= cols)
  q.parallel_for(sycl::range<2>{rows, ld}, [=](sycl::id<2> idx) {
    const std::size_t i = idx[0];  // row
    const std::size_t j = idx[1];  // column (fastest-varying dimension)
    ptr[i * ld + j] = (j < cols) ? gen_value(seed, id, i, j) : nan;
  });
}

// count consecutive doubles set to NaN
void fill_nan(sycl::queue& q, double* ptr, std::size_t count) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  q.parallel_for(sycl::range<1>{count}, [=](sycl::id<1> idx) { ptr[idx[0]] = nan; });
}

} // namespace

Arena::Arena(sycl::queue& q, const Problem& p, std::size_t workspace_bytes) : q_(q), p_(p) {
  if (workspace_bytes % sizeof(double) != 0)
    throw std::invalid_argument("workspace size must be a multiple of sizeof(double)");

  // Offsets and sizes of A, B, C, workspace relative to the arena base
  const ArenaLayout layout = compute_layout(p, workspace_bytes);
  info_.alloc_kind = "sycl::aligned_alloc_device";
  info_.layout = layout;

  // Single page-aligned device allocation for all regions
  base_ = sycl::aligned_alloc_device(arena_page, layout.total_bytes, q_);
  if (base_ == nullptr)
    throw std::runtime_error(info_.alloc_kind + " failed for " + std::to_string(layout.total_bytes) +
                             " bytes");
  // Do not rely on the implementation honoring the alignment argument
  try {
    check_arena_alignment(base_, info_.alloc_kind);
  } catch (...) {
    sycl::free(base_, q_);  // the destructor does not run if the constructor throws
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

Arena::~Arena() {
  if (base_ != nullptr) sycl::free(base_, q_);
}

void Arena::initialize(std::uint64_t seed) {
  // Independent fill kernels, then wait for all of them
  fill_matrix(q_, m_.a, p_.m, p_.k, p_.lda, MatrixId::A, seed);
  fill_matrix(q_, m_.b, p_.k, p_.n, p_.ldb, MatrixId::B, seed);
  fill_nan(q_, m_.c, p_.m * p_.ldc);
  if (m_.workspace != nullptr)
    fill_nan(q_, static_cast<double*>(m_.workspace), m_.workspace_bytes / sizeof(double));
  q_.wait_and_throw();

  // Read back sampled elements and compare with the host value function.
  // Device memory is not accessed directly from the host: each sample is copied with q.memcpy.
  const auto samples = make_init_samples(p_, seed, init_check_count);
  std::vector<double> values(samples.size());
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const auto& s = samples[i];
    const double* src = s.matrix == InitSample::Matrix::A ? m_.a
                      : s.matrix == InitSample::Matrix::B ? m_.b
                                                          : m_.c;
    q_.memcpy(&values[i], src + s.offset, sizeof(double));
  }
  q_.wait_and_throw();  // all copies done before the values are checked
  check_init_samples(p_, seed, samples, values);
}

void Arena::copy_c_to_host(double* dst) const {
  // Whole C including the row padding (M x ldc elements); blocks until the copy is done
  q_.memcpy(dst, m_.c, info_.layout.c_bytes).wait();
}

} // namespace bench
