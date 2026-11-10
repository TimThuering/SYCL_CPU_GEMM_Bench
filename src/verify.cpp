#include "bench/verify.hpp"

#include <cmath>
#include <ostream>
#include <sstream>
#include <vector>

#include "bench/datagen.hpp"

namespace bench {

namespace {

std::string fmt(double x) {
  std::ostringstream s;
  s.precision(17);
  s << x;
  return s.str();
}

std::string pos(std::size_t i, std::size_t j) {
  return "C(" + std::to_string(i) + ", " + std::to_string(j) + ")";
}

} // namespace

VerifyResult verify(const Config& cfg, const double* c) {
  VerifyResult r;
  if (!cfg.verify) return r;
  r.performed = true;
  r.passed = true;

  const Problem& p = cfg.problem;
  auto fail = [&r](const std::string& msg) {
    if (r.passed) {  // keep the first failure
      r.passed = false;
      r.failure = msg;
    }
  };

  // Row padding (columns n .. ldc-1) must still hold the NaN sentinel
  for (std::size_t i = 0; i < p.m && r.passed; ++i)
    for (std::size_t j = p.n; j < p.ldc; ++j)
      if (!std::isnan(c[i * p.ldc + j])) {
        fail(pos(i, j) + " in the row padding was written (" + fmt(c[i * p.ldc + j]) + ")");
        break;
      }

  // A and B on the host, regenerated from the seed
  std::vector<double> a(p.m * p.k), b(p.k * p.n);
  for (std::size_t i = 0; i < p.m; ++i)
    for (std::size_t kk = 0; kk < p.k; ++kk) a[i * p.k + kk] = gen_value(cfg.seed, MatrixId::A, i, kk);
  for (std::size_t kk = 0; kk < p.k; ++kk)
    for (std::size_t j = 0; j < p.n; ++j) b[kk * p.n + j] = gen_value(cfg.seed, MatrixId::B, kk, j);

  // Reference C (i-k-j loop order)
  std::vector<double> ref(p.m * p.n, 0.0);
  for (std::size_t i = 0; i < p.m; ++i)
    for (std::size_t kk = 0; kk < p.k; ++kk) {
      const double aik = a[i * p.k + kk];
      for (std::size_t j = 0; j < p.n; ++j) ref[i * p.n + j] += aik * b[kk * p.n + j];
    }

  // Compare every element
  for (std::size_t i = 0; i < p.m; ++i)
    for (std::size_t j = 0; j < p.n; ++j) {
      const double got = c[i * p.ldc + j];
      const double expected = ref[i * p.n + j];
      const double err = std::fabs(got - expected);
      if (err > r.max_abs_error || std::isnan(err)) r.max_abs_error = std::isnan(err) ? INFINITY : err;
      // NaN-safe: a NaN error (e.g. unwritten element) fails as well
      if (!(err <= verify_tolerance))
        fail(pos(i, j) + " = " + fmt(got) + ", expected " + fmt(expected) + " (|error| " + fmt(err) + ")");
    }
  r.entries_checked = p.m * p.n;
  return r;
}

void print_verify_result(std::ostream& os, const VerifyResult& r) {
  os << "verify:        ";
  if (!r.performed) {
    os << "skipped\n";
    return;
  }
  os << (r.passed ? "PASSED" : "FAILED") << " (" << r.entries_checked << " entries, max |error| "
     << r.max_abs_error << ", tolerance " << verify_tolerance << ")\n";
  if (!r.passed) os << "  first failure: " << r.failure << "\n";
}

} // namespace bench
