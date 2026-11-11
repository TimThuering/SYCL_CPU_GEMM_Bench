# Build information embedded into the binary.
#
#  - generated/build_config.hpp  configure time: impl, targets, flags, compiler, build host
#  - generated/git_info.hpp      build time (every build): commit, describe, dirty flag
#
# Git is only read (rev-parse / describe / status with --no-optional-locks), never modified.

set(BENCH_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${BENCH_GENERATED_DIR}")

# Full `--version` output of the compiler (acpp prints its configuration as well)
execute_process(
  COMMAND ${CMAKE_CXX_COMPILER} --version
  OUTPUT_VARIABLE BENCH_COMPILER_VERSION_OUTPUT
  ERROR_VARIABLE  BENCH_COMPILER_VERSION_OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE
)

# Build host: hostname, CPU model, logical cores, OS, architecture.
# CPU model fallbacks: CMake (x86 "model name") -> lscpu "Model name" (ARM) -> /proc/cpuinfo "uarch",
# then mvendorid / marchid / mimpid (RISC-V). On RISC-V, CMake reports a placeholder such as
# "1 core 0 MHz Unknown family", which is discarded.
cmake_host_system_information(RESULT BENCH_BUILD_HOST  QUERY HOSTNAME)
cmake_host_system_information(RESULT BENCH_BUILD_CORES QUERY NUMBER_OF_LOGICAL_CORES)
cmake_host_system_information(RESULT BENCH_BUILD_CPU   QUERY PROCESSOR_DESCRIPTION)
cmake_host_system_information(RESULT _os_name          QUERY OS_NAME)
cmake_host_system_information(RESULT _os_release       QUERY OS_RELEASE)
set(BENCH_BUILD_OS   "${_os_name} ${_os_release}")
set(BENCH_BUILD_ARCH "${CMAKE_HOST_SYSTEM_PROCESSOR}")

string(STRIP "${BENCH_BUILD_CPU}" BENCH_BUILD_CPU)
if(BENCH_BUILD_CPU MATCHES "Unknown family")
  set(BENCH_BUILD_CPU "")
endif()
if(BENCH_BUILD_CPU STREQUAL "")
  execute_process(COMMAND lscpu OUTPUT_VARIABLE _lscpu ERROR_QUIET RESULT_VARIABLE _rc)
  if(_rc EQUAL 0 AND _lscpu MATCHES "Model name:[ \t]*([^\n]*)")
    string(STRIP "${CMAKE_MATCH_1}" BENCH_BUILD_CPU)
  endif()
endif()
if(BENCH_BUILD_CPU STREQUAL "" AND EXISTS "/proc/cpuinfo")
  file(STRINGS "/proc/cpuinfo" _uarch REGEX "^uarch[ \t]*:" LIMIT_COUNT 1)
  if(_uarch MATCHES ":[ \t]*(.*)$")
    string(STRIP "${CMAKE_MATCH_1}" BENCH_BUILD_CPU)
  endif()
endif()
if(BENCH_BUILD_CPU STREQUAL "" AND EXISTS "/proc/cpuinfo")
  set(_ids "")
  foreach(_key mvendorid marchid mimpid)
    file(STRINGS "/proc/cpuinfo" _line REGEX "^${_key}[ \t]*:" LIMIT_COUNT 1)
    if(_line MATCHES ":[ \t]*(.*)$")
      string(STRIP "${CMAKE_MATCH_1}" _val)
      list(APPEND _ids "${_key} ${_val}")
    endif()
  endforeach()
  list(JOIN _ids ", " BENCH_BUILD_CPU)
endif()
if(BENCH_BUILD_CPU STREQUAL "")
  set(BENCH_BUILD_CPU "unknown")
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt_upper)
set(BENCH_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_bt_upper}}")
string(STRIP "${BENCH_CXX_FLAGS}" BENCH_CXX_FLAGS)

# Escape values for use inside C string literals: ESC_<var>
function(bench_escape_c var)
  set(_v "${${var}}")
  string(REPLACE "\\" "\\\\" _v "${_v}")
  string(REPLACE "\"" "\\\"" _v "${_v}")
  string(REPLACE "\r" ""     _v "${_v}")
  string(REPLACE "\n" "\\n"  _v "${_v}")
  set(ESC_${var} "${_v}" PARENT_SCOPE)
endfunction()

foreach(_var SYCL_IMPL SYCL_TARGETS SYCL_EXTRA_FLAGS ARCH_FLAGS BENCH_EFFECTIVE_FLAGS BENCH_CXX_FLAGS
             CMAKE_BUILD_TYPE CMAKE_CXX_COMPILER CMAKE_CXX_COMPILER_ID CMAKE_CXX_COMPILER_VERSION
             BENCH_COMPILER_VERSION_OUTPUT CMAKE_VERSION
             BENCH_BUILD_HOST BENCH_BUILD_CPU BENCH_BUILD_CORES BENCH_BUILD_OS BENCH_BUILD_ARCH)
  bench_escape_c(${_var})
endforeach()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_config.hpp.in"
  "${BENCH_GENERATED_DIR}/build_config.hpp"
  @ONLY
)

# Git info: regenerated on every build, file only rewritten when the content changes
add_custom_target(bench_git_info ALL
  COMMAND ${CMAKE_COMMAND}
          -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}
          -DOUT_FILE=${BENCH_GENERATED_DIR}/git_info.hpp
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GitInfo.cmake
  BYPRODUCTS ${BENCH_GENERATED_DIR}/git_info.hpp
  COMMENT "Reading git state (read-only)"
  VERBATIM
)
