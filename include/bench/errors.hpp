#pragma once

#include <stdexcept>

namespace bench {

// The selected stage cannot run in this build (result status UNSUPPORTED, exit code 4)
struct UnsupportedError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

} // namespace bench
