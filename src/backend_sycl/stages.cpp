#include "stages.hpp"

#include <array>

#include "sycl_stages/interchange.hpp"
#include "sycl_stages/naive.hpp"
#include "sycl_stages/packed.hpp"
#include "sycl_stages/ref.hpp"
#include "sycl_stages/regblock.hpp"

namespace bench {

namespace {

// Names as in include/bench/stages.hpp
const std::array<SyclStage, 5> stage_table{{
    {"ref", kernels::ref::workspace_bytes, kernels::ref::run},
    {"naive", kernels::naive::workspace_bytes, kernels::naive::run},
    {"interchange", kernels::interchange::workspace_bytes, kernels::interchange::run},
    {"regblock", kernels::regblock::workspace_bytes, kernels::regblock::run},
    {"packed", kernels::packed::workspace_bytes, kernels::packed::run},
}};

} // namespace

const SyclStage* find_stage(std::string_view name) {
  for (const auto& s : stage_table)
    if (s.name == name) return &s;
  return nullptr;
}

} // namespace bench
