#pragma once
// Table of the OpenCL reference stages: name -> workspace size.

#include <cstddef>
#include <string_view>

#include "bench/config.hpp"

namespace bench {

struct ClStage {
  std::string_view name;
  std::size_t (*workspace_bytes)(const Problem&);
};

// nullptr if no stage with this name is registered
const ClStage* find_stage(std::string_view name);

} // namespace bench
