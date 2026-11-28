#pragma once
// Table of the OpenCL reference stages: name -> workspace size, kernel source, run function.
// Per stage: kernels/opencl_stages/<name>.cl (OpenCL C kernels, embedded as bench::cl_source::<name>)
// and kernels/opencl_stages/<name>.hpp (host part: workspace size, argument setting, enqueue).

#include <cstddef>
#include <string_view>

#include "backend_opencl/gemm_args.hpp"
#include "backend_opencl/program.hpp"
#include "bench/config.hpp"

namespace bench {

struct ClStage {
  std::string_view name;
  std::size_t (*workspace_bytes)(const Problem&);
  const char* source;                                // OpenCL C source of the stage's kernels
  void (*run)(ClStageProgram&, const ClGemmArgs&);   // enqueues the work; the caller waits
};

// nullptr if no stage with this name is registered
const ClStage* find_stage(std::string_view name);

} // namespace bench
