#include "stages.hpp"

#include <array>

namespace bench {

namespace {

// Names as in include/bench/stages.hpp
const std::array<ClStage, 0> stage_table{};

} // namespace

const ClStage* find_stage(std::string_view name) {
  for (const auto& s : stage_table)
    if (s.name == name) return &s;
  return nullptr;
}

} // namespace bench
