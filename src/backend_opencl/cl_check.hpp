#pragma once
// OpenCL C API (CL_TARGET_OPENCL_VERSION is set by CMake) and the error check used for every call.

#include <CL/cl.h>
#include <CL/cl_ext.h>  // CL_PLATFORM_NOT_FOUND_KHR (ICD loader)

#include <stdexcept>
#include <string>

namespace bench {

// Throws std::runtime_error naming the failed call and the OpenCL error code (CL/cl.h)
inline void cl_check(cl_int err, const char* call) {
  if (err != CL_SUCCESS)
    throw std::runtime_error(std::string(call) + " failed (OpenCL error " + std::to_string(err) + ")");
}

} // namespace bench
