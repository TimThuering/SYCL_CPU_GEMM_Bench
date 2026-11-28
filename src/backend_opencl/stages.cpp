#include "stages.hpp"

#include <array>

#include "cl_sources/interchange.cl.hpp"
#include "cl_sources/naive.cl.hpp"
#include "cl_sources/packed.cl.hpp"
#include "cl_sources/ref.cl.hpp"
#include "cl_sources/regblock.cl.hpp"
#include "opencl_stages/interchange.hpp"
#include "opencl_stages/naive.hpp"
#include "opencl_stages/packed.hpp"
#include "opencl_stages/ref.hpp"
#include "opencl_stages/regblock.hpp"

namespace bench {

namespace {

namespace k = kernels::opencl;

// Names as in include/bench/stages.hpp
const std::array<ClStage, 5> stage_table{{
    {"ref", k::ref::workspace_bytes, cl_source::ref, k::ref::run},
    {"naive", k::naive::workspace_bytes, cl_source::naive, k::naive::run},
    {"interchange", k::interchange::workspace_bytes, cl_source::interchange, k::interchange::run},
    {"regblock", k::regblock::workspace_bytes, cl_source::regblock, k::regblock::run},
    {"packed", k::packed::workspace_bytes, cl_source::packed, k::packed::run},
}};

} // namespace

const ClStage* find_stage(std::string_view name) {
  for (const auto& s : stage_table)
    if (s.name == name) return &s;
  return nullptr;
}

} // namespace bench
