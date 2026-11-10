cmake_minimum_required(VERSION 3.20)

project(SYCL_CPU_GEMM_Bench LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Build configuration
#
#   SYCL_IMPL        acpp | dpcpp | none   (none = plain C++/OpenMP twin)
#   SYCL_TARGETS     acpp:  value for --acpp-targets   (e.g. omp.accelerated, generic)
#                    dpcpp: value for -fsycl-targets   (e.g. native_cpu, spir64)
#   SYCL_EXTRA_FLAGS additional SYCL compiler flags (compile + link), e.g. device flags
#   ARCH_FLAGS       host ISA flags (compile + link), e.g. "-mcpu=native" or "-march=..."
#   GEMM_MR/NR/KC/MC compile-time kernel parameters (see kernels/params.hpp)
#
# The compiler itself is chosen with CMAKE_CXX_COMPILER (acpp, DPC++ clang++, clang++).
# See docs/building.md for one configure line per flow.
# -----------------------------------------------------------------------------

set(SYCL_IMPL "none" CACHE STRING "SYCL implementation: acpp | dpcpp | none")
set_property(CACHE SYCL_IMPL PROPERTY STRINGS acpp dpcpp none)
set(SYCL_TARGETS "" CACHE STRING "SYCL targets (acpp: --acpp-targets, dpcpp: -fsycl-targets)")
set(SYCL_EXTRA_FLAGS "" CACHE STRING "Additional SYCL compiler flags (compile + link)")
set(ARCH_FLAGS "" CACHE STRING "Host ISA flags (compile + link)")

set(GEMM_MR 4   CACHE STRING "Micro-kernel rows (register block)")
set(GEMM_NR 8   CACHE STRING "Micro-kernel columns (register block)")
set(GEMM_KC 256 CACHE STRING "Cache block size along K")
set(GEMM_MC 64  CACHE STRING "Cache block size along M")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(Dependencies)   # nlohmann_json, cxxopts (FetchContent)
include(SyclFlags)      # interface target bench_flags (impl, targets, arch, params)
include(BuildInfo)      # generated/build_config.hpp + build-time generated/git_info.hpp

# -----------------------------------------------------------------------------
# Harness core: SYCL-free code shared by the SYCL driver and the OpenMP twin
# -----------------------------------------------------------------------------
add_library(bench_core STATIC
  src/build_info.cpp
  src/cli.cpp
  src/config.cpp
  src/init_check.cpp
  src/layout.cpp
  src/metadata.cpp
  src/result.cpp
  src/timing.cpp
  src/verify.cpp
)
target_include_directories(bench_core
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
          ${CMAKE_CURRENT_SOURCE_DIR}/kernels
  PRIVATE ${BENCH_GENERATED_DIR}
)
target_link_libraries(bench_core PUBLIC bench_flags nlohmann_json::nlohmann_json cxxopts::cxxopts)
add_dependencies(bench_core bench_git_info)

# -----------------------------------------------------------------------------
# Driver: one executable name for every flow, the source depends on SYCL_IMPL
# -----------------------------------------------------------------------------
if(SYCL_IMPL STREQUAL "none")
  add_executable(gemm_bench
    src/main_omp.cpp
    src/backend_omp/arena.cpp
    src/backend_omp/device.cpp
    src/backend_omp/stages.cpp
  )
else()
  add_executable(gemm_bench
    src/main_sycl.cpp
    src/backend_sycl/arena.cpp
    src/backend_sycl/device.cpp
    src/backend_sycl/stages.cpp
  )
endif()
target_link_libraries(gemm_bench PRIVATE bench_core)

message(STATUS "gemm_bench: SYCL_IMPL=${SYCL_IMPL} SYCL_TARGETS='${SYCL_TARGETS}' "
               "ARCH_FLAGS='${ARCH_FLAGS}' SYCL_EXTRA_FLAGS='${SYCL_EXTRA_FLAGS}' "
               "MR=${GEMM_MR} NR=${GEMM_NR} KC=${GEMM_KC} MC=${GEMM_MC} "
               "BUILD_TYPE=${CMAKE_BUILD_TYPE}")
