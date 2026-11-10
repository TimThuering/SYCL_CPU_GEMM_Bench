# cmake -DBIN=<gemm_bench> -DEXPECTED=<code> -DARGS="<args>" -P expect_exit_code.cmake
# Runs the binary with ARGS and fails unless it exits with EXPECTED.

separate_arguments(ARGS)
execute_process(COMMAND ${BIN} ${ARGS} RESULT_VARIABLE rc OUTPUT_QUIET ERROR_QUIET)
if(NOT rc EQUAL EXPECTED)
  message(FATAL_ERROR "${BIN} ${ARGS}: exit code ${rc}, expected ${EXPECTED}")
endif()
