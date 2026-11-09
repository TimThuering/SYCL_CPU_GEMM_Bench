# Interface target `bench_flags`: everything that depends on the flow.
# Linked by all project targets, so every translation unit sees the same flags.

add_library(bench_flags INTERFACE)

separate_arguments(_arch_flags  UNIX_COMMAND "${ARCH_FLAGS}")
separate_arguments(_extra_flags UNIX_COMMAND "${SYCL_EXTRA_FLAGS}")

set(_sycl_flags "")
if(SYCL_IMPL STREQUAL "acpp")
  if(SYCL_TARGETS STREQUAL "")
    message(FATAL_ERROR "SYCL_IMPL=acpp requires SYCL_TARGETS (e.g. omp.accelerated, omp.library-only, generic)")
  endif()
  if(NOT CMAKE_CXX_COMPILER MATCHES "acpp")
    message(WARNING "SYCL_IMPL=acpp but CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} does not look like acpp")
  endif()
  set(_sycl_flags "--acpp-targets=${SYCL_TARGETS}")
  target_compile_definitions(bench_flags INTERFACE BENCH_SYCL=1 BENCH_IMPL_ACPP=1)

elseif(SYCL_IMPL STREQUAL "dpcpp")
  if(SYCL_TARGETS STREQUAL "")
    message(FATAL_ERROR "SYCL_IMPL=dpcpp requires SYCL_TARGETS (e.g. native_cpu, spir64)")
  endif()
  set(_sycl_flags "-fsycl" "-fsycl-targets=${SYCL_TARGETS}")
  target_compile_definitions(bench_flags INTERFACE BENCH_SYCL=1 BENCH_IMPL_DPCPP=1)

elseif(SYCL_IMPL STREQUAL "none")
  if(NOT SYCL_TARGETS STREQUAL "")
    message(WARNING "SYCL_TARGETS is ignored for SYCL_IMPL=none")
  endif()
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(bench_flags INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(bench_flags INTERFACE BENCH_SYCL=0 BENCH_IMPL_NONE=1)

else()
  message(FATAL_ERROR "Unknown SYCL_IMPL='${SYCL_IMPL}' (expected acpp | dpcpp | none)")
endif()

target_compile_options(bench_flags INTERFACE ${_sycl_flags} ${_extra_flags} ${_arch_flags})
target_link_options(bench_flags    INTERFACE ${_sycl_flags} ${_extra_flags} ${_arch_flags})

target_compile_definitions(bench_flags INTERFACE
  GEMM_MR=${GEMM_MR}
  GEMM_NR=${GEMM_NR}
  GEMM_KC=${GEMM_KC}
  GEMM_MC=${GEMM_MC}
)

# Effective flag string, recorded in the build info
string(JOIN " " BENCH_EFFECTIVE_FLAGS ${_sycl_flags} ${_extra_flags} ${_arch_flags})
