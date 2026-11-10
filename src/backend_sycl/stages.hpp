#pragma once
// Table of the SYCL stages: name -> workspace size + run function (kernels/sycl_stages/*.hpp).

#include <sycl/sycl.hpp>

#include <cstddef>
#include <string_view>

#include "bench/config.hpp"
#include "gemm_args.hpp"

namespace bench {

struct SyclStage {
  std::string_view name;
  std::size_t (*workspace_bytes)(const Problem&);
  void (*run)(sycl::queue&, const GemmArgs&);  // submits the work; the caller waits
};

// nullptr if no stage with this name is registered
const SyclStage* find_stage(std::string_view name);

} // namespace bench
