#pragma once
// Table of the OpenMP twin stages: name -> workspace size + run function (kernels/omp_stages/*.hpp).

#include <cstddef>
#include <string_view>

#include "bench/config.hpp"
#include "gemm_args.hpp"

namespace bench {

struct OmpStage {
  std::string_view name;
  std::size_t (*workspace_bytes)(const Problem&);
  void (*run)(const GemmArgs&);  // returns when C is computed
};

// nullptr if no stage with this name is registered
const OmpStage* find_stage(std::string_view name);

} // namespace bench
