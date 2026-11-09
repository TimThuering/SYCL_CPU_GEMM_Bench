#include "bench/config.hpp"

#include <ostream>

namespace bench {

std::string to_string(VerifyMode mode) {
  switch (mode) {
    case VerifyMode::fast: return "fast";
    case VerifyMode::full: return "full";
    case VerifyMode::none: return "none";
  }
  return "unknown";
}

void print_config(std::ostream& os, const Config& cfg) {
  const Problem& p = cfg.problem;
  os << "stage:         " << cfg.stage << "\n"
     << "problem:       M=" << p.m << " N=" << p.n << " K=" << p.k
     << " (lda=" << p.lda << " ldb=" << p.ldb << " ldc=" << p.ldc << ")\n"
     << "seed:          " << cfg.seed << "\n"
     << "iterations:    1 cold + " << cfg.warmup << " warmup + " << cfg.reps << " timed"
     << (cfg.profiling ? " (event profiling)" : "") << "\n"
     << "verify:        " << to_string(cfg.verify);
  if (cfg.verify == VerifyMode::fast) os << " (" << cfg.verify_samples << " samples)";
  os << "\n"
     << "output:        " << (cfg.output.empty() ? "stdout" : cfg.output) << "\n";
  if (!cfg.label.empty()) os << "label:         " << cfg.label << "\n";
}

} // namespace bench
