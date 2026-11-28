# Script mode (cmake -DIN=... -DOUT=... -DNAME=... -P embed_cl_source.cmake), called by bench_embed_cl:
# writes the content of IN as a C++ raw string literal bench::cl_source::NAME into OUT.

file(READ "${IN}" _src)
if(_src MATCHES "\\)CLSRC\"")
  message(FATAL_ERROR "${IN} contains the raw string delimiter )CLSRC\"")
endif()

file(WRITE "${OUT}" "#pragma once
// Generated from ${IN} by cmake/embed_cl_source.cmake - do not edit.

namespace bench::cl_source {

inline constexpr const char ${NAME}[] = R\"CLSRC(${_src})CLSRC\";

} // namespace bench::cl_source
")
