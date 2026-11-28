# Embedding of OpenCL C kernel sources (only for SYCL_IMPL=none_opencl).
#
#   bench_embed_cl(<target> <file.cl>)
#
# Generates ${BENCH_GENERATED_DIR}/cl_sources/<name>.cl.hpp with the file content as the string
# bench::cl_source::<name> (<name> = file name without .cl), regenerated whenever the .cl file
# changes, and adds it to <target>. The OpenCL driver compiles the string at run time
# (clBuildProgram), so the binary does not depend on the .cl files at run time.

function(bench_embed_cl target file)
  get_filename_component(_name "${file}" NAME_WE)
  get_filename_component(_in "${file}" ABSOLUTE)
  set(_out "${BENCH_GENERATED_DIR}/cl_sources/${_name}.cl.hpp")
  add_custom_command(
    OUTPUT  "${_out}"
    COMMAND ${CMAKE_COMMAND} -DIN=${_in} -DOUT=${_out} -DNAME=${_name}
            -P ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/embed_cl_source.cmake
    DEPENDS "${_in}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/embed_cl_source.cmake"
    COMMENT "Embedding OpenCL source ${_name}.cl"
    VERBATIM
  )
  target_sources(${target} PRIVATE "${_out}")
  target_include_directories(${target} PRIVATE "${BENCH_GENERATED_DIR}")
endfunction()
