// Packing kernels of the packed stage (OpenMP twin): every slot of Ap / Bp against the layout formulas,
// incl. zero padding, the shorter last panel and slots beyond kc (must stay untouched).

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "omp_stages/packed.hpp"
#include "packing_test_cases.hpp"

namespace {

using namespace kernels::omp::packed;
using namespace packing_tests;

TEST(PackingOmp, AllSlots) {
  for (const Shape& sh : shapes()) {
    const std::size_t lda = sh.k + sh.pad, ldb = sh.n + sh.pad;
    const auto a = make_matrix(sh.m, sh.k, lda);
    const auto b = make_matrix(sh.k, sh.n, ldb);
    const std::size_t rows_p = ceil_div(sh.m, MR) * MR, cols_p = ceil_div(sh.n, NR) * NR;

    for (std::size_t pc = 0; pc < sh.k; pc += KC) {
      const std::size_t kc = std::min(KC, sh.k - pc);
      std::vector<double> ap(rows_p * KC, untouched), bp(cols_p * KC, untouched);
      pack_b(b.data(), ldb, sh.n, pc, kc, bp.data());
      pack_a(a.data(), lda, sh.m, pc, kc, ap.data());

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
    }
  }
}

} // namespace
