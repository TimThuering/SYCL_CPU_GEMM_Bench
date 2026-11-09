#include "bench/build_info.hpp"

#include "build_config.hpp"  // generated at configure time
#include "git_info.hpp"      // generated at build time
#include "params.hpp"

#include <ostream>

namespace bench {

const BuildInfo& build_info() {
  static const BuildInfo info{
      BENCH_CFG_SYCL_IMPL,
      BENCH_CFG_SYCL_TARGETS,
      BENCH_CFG_SYCL_EXTRA_FLAGS,
      BENCH_CFG_ARCH_FLAGS,
      BENCH_CFG_EFFECTIVE_FLAGS,
      BENCH_CFG_CXX_FLAGS,
      BENCH_CFG_BUILD_TYPE,
      BENCH_CFG_COMPILER_PATH,
      BENCH_CFG_COMPILER_ID,
      BENCH_CFG_COMPILER_VERSION,
      BENCH_CFG_COMPILER_VERSION_OUTPUT,
      BENCH_CFG_CMAKE_VERSION,
      BENCH_CFG_BUILD_HOST,
      BENCH_CFG_BUILD_CPU,
      BENCH_CFG_BUILD_CORES,
      BENCH_CFG_BUILD_OS,
      BENCH_CFG_BUILD_ARCH,
      BENCH_GIT_COMMIT,
      BENCH_GIT_DESCRIBE,
      BENCH_GIT_DIRTY,
      gemm::MR,
      gemm::NR,
      gemm::KC,
      gemm::MC,
  };
  return info;
}

void print_build_info(std::ostream& os) {
  const auto& b = build_info();
  os << "impl:          " << b.sycl_impl << "\n"
     << "targets:       " << b.sycl_targets << "\n"
     << "flags:         " << b.effective_flags << "\n"
     << "cxx flags:     " << b.cxx_flags << " (" << b.build_type << ")\n"
     << "compiler:      " << b.compiler_id << " " << b.compiler_version << " (" << b.compiler_path << ")\n"
     << "build host:    " << b.build_host << " (" << b.build_cpu << ", " << b.build_cores << " logical cores, "
     << b.build_arch << ", " << b.build_os << ")\n"
     << "git:           " << b.git_describe << " commit=" << b.git_commit << " dirty=" << b.git_dirty << "\n"
     << "params:        MR=" << b.mr << " NR=" << b.nr << " KC=" << b.kc << " MC=" << b.mc << "\n";
}

} // namespace bench
