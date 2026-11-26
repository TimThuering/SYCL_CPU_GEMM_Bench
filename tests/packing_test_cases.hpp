#pragma once
// Expected contents of the packed buffers of the packed stage (shared by the SYCL and the OpenMP test).
// Matrix values encode their position (A(i, j) = 1000 i + j), so a wrong index shows as a wrong value.

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace packing_tests {

struct Shape {
  std::size_t m, n, k, pad;
};

inline std::vector<Shape> shapes() {
  return {{3, 3, 3, 0}, {5, 7, 4, 0}, {33, 17, 600, 3}, {1, 1, 1, 0}, {8, 16, 256, 0}, {9, 9, 257, 2}};
}

// rows x ld matrix with value 1000 * row + col, NaN in the row padding
inline std::vector<double> make_matrix(std::size_t rows, std::size_t cols, std::size_t ld) {
  std::vector<double> x(rows * ld);
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < ld; ++j) x[i * ld + j] = j < cols ? 1000.0 * i + j : NAN;
  return x;
}

inline double value(std::size_t row, std::size_t col) { return 1000.0 * row + col; }

constexpr double untouched = -1.0;  // buffer fill before packing; slots with p >= kc must keep it

} // namespace packing_tests
