#include "stages.hpp"

#include <array>

#include "omp_stages/interchange.hpp"
#include "omp_stages/naive.hpp"
#include "omp_stages/packed.hpp"
#include "omp_stages/ref.hpp"
#include "omp_stages/regblock.hpp"

namespace bench {

namespace {

// Names as in include/bench/stages.hpp
const std::array<OmpStage, 5> stage_table{{
    {"ref", kernels::omp::ref::workspace_bytes, kernels::omp::ref::run},
    {"naive", kernels::omp::naive::workspace_bytes, kernels::omp::naive::run},
    {"interchange", kernels::omp::interchange::workspace_bytes, kernels::omp::interchange::run},
    {"regblock", kernels::omp::regblock::workspace_bytes, kernels::omp::regblock::run},
    {"packed", kernels::omp::packed::workspace_bytes, kernels::omp::packed::run},
}};

} // namespace

const OmpStage* find_stage(std::string_view name) {
  for (const auto& s : stage_table)
    if (s.name == name) return &s;
  return nullptr;
}

} // namespace bench
