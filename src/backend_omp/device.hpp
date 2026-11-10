#pragma once
// "Device" information for the OpenMP twin: the host with its OpenMP configuration.

#include "bench/metadata.hpp"

namespace bench {

DeviceInfo query_device();

} // namespace bench
