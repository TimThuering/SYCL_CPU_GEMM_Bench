# GoogleTest tests (only with -DBENCH_BUILD_TESTS=ON)
#
#   bench_unit_tests   harness core (SYCL-free): data, layout, CLI, verification, timing, JSON
#   bench_stage_tests  every stage of this build (SYCL or OpenMP twin) on odd sizes, verified
#   cli_*              end-to-end runs of the gemm_bench binary (exit codes)

include(FetchContent)
set(INSTALL_GTEST OFF CACHE INTERNAL "")
set(BUILD_GMOCK   OFF CACHE INTERNAL "")
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_executable(bench_unit_tests
  test_cli.cpp
  test_data.cpp
  test_result.cpp
  test_timing.cpp
  test_verify.cpp
)
target_link_libraries(bench_unit_tests PRIVATE bench_core GTest::gtest_main)

add_executable(bench_stage_tests test_stages_${BENCH_BACKEND}.cpp)
target_link_libraries(bench_stage_tests PRIVATE bench_backend GTest::gtest_main)

# PRE_TEST: tests are listed when ctest runs, not at build time (no SYCL runtime start during the build)
gtest_discover_tests(bench_unit_tests  DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(bench_stage_tests DISCOVERY_MODE PRE_TEST)

# End-to-end: the reference stage runs and verifies (exit 0), an invalid command line gives exit 2
add_test(NAME cli_ref
         COMMAND gemm_bench -s ref -m 33 -n 17 -k 21 --ld-pad 3 --reps 2 -q
                 -o ${CMAKE_CURRENT_BINARY_DIR}/cli_ref.json)
add_test(NAME cli_usage_exit_code
         COMMAND ${CMAKE_COMMAND} -DBIN=$<TARGET_FILE:gemm_bench> -DEXPECTED=2 "-DARGS=-s ref --bogus"
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_exit_code.cmake)
