// Packing kernels of the packed stage (SYCL): every slot of Ap / Bp against the layout formulas,
// incl. zero padding, the shorter last panel and slots beyond kc (must stay untouched).
// Uses USM shared memory, so the host can fill the inputs and read the packed buffers directly.

#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <vector>

#include "backend_sycl/queue.hpp"
#include "packing_test_cases.hpp"
#include "sycl_stages/packed.hpp"

namespace {

using namespace kernels::packed;
using namespace packing_tests;

// USM shared copy of a host vector (freed by the caller)
double* to_shared(sycl::queue& q, const std::vector<double>& v) {
  double* p = sycl::malloc_shared<double>(v.size(), q);
  std::copy(v.begin(), v.end(), p);
  return p;
}

TEST(PackingSycl, AllSlots) {
  sycl::queue q = bench::make_queue();
  for (const Shape& sh : shapes()) {
    const std::size_t lda = sh.k + sh.pad, ldb = sh.n + sh.pad;
    double* a = to_shared(q, make_matrix(sh.m, sh.k, lda));
    double* b = to_shared(q, make_matrix(sh.k, sh.n, ldb));
    const std::size_t rows_p = ceil_div(sh.m, MR) * MR, cols_p = ceil_div(sh.n, NR) * NR;

    for (std::size_t pc = 0; pc < sh.k; pc += KC) {
      const std::size_t kc = std::min(KC, sh.k - pc);
      double* ap = to_shared(q, std::vector<double>(rows_p * KC, untouched));
      double* bp = to_shared(q, std::vector<double>(cols_p * KC, untouched));
      pack_b(q, b, ldb, sh.n, pc, kc, bp);
      pack_a(q, a, lda, sh.m, pc, kc, ap);
      q.wait_and_throw();

      for (std::size_t jr = 0; jr < cols_p / NR; ++jr)
        for (std::size_t p = 0; p < KC; ++p)
          for (std::size_t s = 0; s < NR; ++s) {
            const std::size_t j = jr * NR + s;
            const double expected = p >= kc ? untouched : (j < sh.n ? value(pc + p, j) : 0.0);
            ASSERT_EQ(bp[(jr * KC + p) * NR + s], expected)
                << "Bp shape " << sh.m << "x" << sh.n << "x" << sh.k << " pc=" << pc << " jr=" << jr << " p=" << p << " s=" << s;
          }
      for (std::size_t ir = 0; ir < rows_p / MR; ++ir)
        for (std::size_t p = 0; p < KC; ++p)
          for (std::size_t r = 0; r < MR; ++r) {
            const std::size_t i = ir * MR + r;
            const double expected = p >= kc ? untouched : (i < sh.m ? value(i, pc + p) : 0.0);
            ASSERT_EQ(ap[(ir * KC + p) * MR + r], expected)
                << "Ap shape " << sh.m << "x" << sh.n << "x" << sh.k << " pc=" << pc << " ir=" << ir << " p=" << p << " r=" << r;
          }
      sycl::free(ap, q);
      sycl::free(bp, q);
    }
    sycl::free(a, q);
    sycl::free(b, q);
  }
}

} // namespace
