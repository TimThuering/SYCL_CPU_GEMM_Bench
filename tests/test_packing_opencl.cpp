// Packing kernels of the packed stage (OpenCL reference): every slot of Ap / Bp against the layout
// formulas, incl. zero padding, the shorter last panel and slots beyond kc (must stay untouched).
// Inputs and the workspace are buffers initialized from host vectors; the workspace is read back.

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "backend_opencl/handle.hpp"
#include "backend_opencl/program.hpp"
#include "backend_opencl/runtime.hpp"
#include "cl_sources/packed.cl.hpp"
#include "opencl_stages/packed.hpp"
#include "packing_test_cases.hpp"

namespace {

using namespace kernels::opencl::packed;
using namespace packing_tests;

// Buffer initialized with a copy of a host vector
bench::ClMem to_buffer(const bench::ClRuntime& rt, std::vector<double> v) {
  cl_int err = CL_SUCCESS;
  bench::ClMem buf(clCreateBuffer(rt.context.get(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  v.size() * sizeof(double), v.data(), &err));
  bench::cl_check(err, "clCreateBuffer");
  return buf;
}

TEST(PackingOpenCl, AllSlots) {
  const bench::ClRuntime rt = bench::make_runtime();
  bench::ClStageProgram prog(rt, bench::cl_source::packed);
  for (const Shape& sh : shapes()) {
    const std::size_t lda = sh.k + sh.pad, ldb = sh.n + sh.pad;
    const bench::ClMem a = to_buffer(rt, make_matrix(sh.m, sh.k, lda));
    const bench::ClMem b = to_buffer(rt, make_matrix(sh.k, sh.n, ldb));
    const std::size_t rows_p = ceil_div(sh.m, MR) * MR, cols_p = ceil_div(sh.n, NR) * NR;
    const std::size_t bp_offset = rows_p * KC;

    for (std::size_t pc = 0; pc < sh.k; pc += KC) {
      const std::size_t kc = std::min(KC, sh.k - pc);
      const bench::ClMem ws = to_buffer(rt, std::vector<double>((rows_p + cols_p) * KC, untouched));
      pack_b(prog, b.get(), ldb, sh.n, pc, kc, ws.get(), bp_offset);
      pack_a(prog, a.get(), lda, sh.m, pc, kc, ws.get());
      ASSERT_NO_THROW(bench::finish(rt));
      std::vector<double> w((rows_p + cols_p) * KC);
      ASSERT_EQ(clEnqueueReadBuffer(rt.queue.get(), ws.get(), CL_TRUE, 0, w.size() * sizeof(double), w.data(), 0,
                                    nullptr, nullptr),
                CL_SUCCESS);
      const double* ap = w.data();
      const double* bp = w.data() + bp_offset;

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
