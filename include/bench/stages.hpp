#pragma once
// Names of the selectable kernel stages.

#include <array>
#include <string_view>

namespace bench {

struct StageName {
  std::string_view name;
  std::string_view description;
};

inline constexpr std::array<StageName, 5> stage_names{{
    {"ref",         "reference: single_task triple loop (harness testing only, small sizes)"},
    {"naive",       "S1: one C element per work-item"},
    {"interchange", "S2: one C row per work-item, k outer / j inner"},
    {"regblock",    "S3: MR x NR register block per work-item"},
    {"packed",      "S4: packed panels + KC/MC cache blocking around the S3 micro-kernel"},
}};

inline bool is_stage_name(std::string_view name) {
  for (const auto& s : stage_names)
    if (s.name == name) return true;
  return false;
}

} // namespace bench
