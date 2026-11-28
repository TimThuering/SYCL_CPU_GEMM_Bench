#pragma once
// Owning handle for OpenCL objects: releases the object when it goes out of scope (move-only).

#include <utility>

#include "backend_opencl/cl_check.hpp"

namespace bench {

template <typename T, cl_int (*Release)(T)>
class ClHandle {
public:
  ClHandle() = default;
  explicit ClHandle(T h) : h_(h) {}
  ~ClHandle() { reset(); }

  ClHandle(const ClHandle&) = delete;
  ClHandle& operator=(const ClHandle&) = delete;
  ClHandle(ClHandle&& o) noexcept : h_(std::exchange(o.h_, nullptr)) {}
  ClHandle& operator=(ClHandle&& o) noexcept {
    if (this != &o) {
      reset();
      h_ = std::exchange(o.h_, nullptr);
    }
    return *this;
  }

  T get() const { return h_; }

private:
  void reset() {
    if (h_ != nullptr) Release(h_);  // error code ignored: nothing sensible to do in a destructor
    h_ = nullptr;
  }

  T h_ = nullptr;
};

using ClContext = ClHandle<cl_context, clReleaseContext>;
using ClQueue = ClHandle<cl_command_queue, clReleaseCommandQueue>;

} // namespace bench
