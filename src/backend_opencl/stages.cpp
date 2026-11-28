#include "stages.hpp"

#include <array>

#include "cl_sources/ref.cl.hpp"
#include "opencl_stages/ref.hpp"

namespace bench {

namespace {

// Names as in include/bench/stages.hpp
const std::array<ClStage, 1> stage_table{{
    {"ref", kernels::opencl::ref::workspace_bytes, cl_source::ref, kernels::opencl::ref::run},
}};

} // namespace

const ClStage* find_stage(std::string_view name) {
  for (const auto& s : stage_table)
    if (s.name == name) return &s;
  return nullptr;
}

} // namespace bench
