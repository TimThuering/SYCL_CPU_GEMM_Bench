#pragma once
// Compile-time kernel parameters (fixed for the study, not tuned per run).
// Set via CMake (-DGEMM_MR=... etc.); the defaults below only apply if the
// header is used outside the CMake build.
//
//   MR x NR  register block (micro-kernel), S3/S4
//   KC       cache block along K (B micro-panel KC x NR should fit ~half of L1), S4
//   MC       cache block along M (packed A block MC x KC should fit the per-core L2 share), S4
//

#include <cstddef>

#ifndef GEMM_MR
#define GEMM_MR 4
#endif
#ifndef GEMM_NR
#define GEMM_NR 8
#endif
#ifndef GEMM_KC
#define GEMM_KC 256
#endif
#ifndef GEMM_MC
#define GEMM_MC 64
#endif

namespace gemm {

inline constexpr std::size_t MR = GEMM_MR;
inline constexpr std::size_t NR = GEMM_NR;
inline constexpr std::size_t KC = GEMM_KC;
inline constexpr std::size_t MC = GEMM_MC;

static_assert(MR > 0 && NR > 0 && KC > 0 && MC > 0, "kernel parameters must be positive");
static_assert(MC % MR == 0, "MC must be a multiple of MR");

} // namespace gemm
