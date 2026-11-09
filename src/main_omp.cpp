// Plain C++/OpenMP twin driver (SYCL_IMPL=none).

#include <omp.h>

#include <iostream>

#include "bench/build_info.hpp"

int main() {
  bench::print_build_info(std::cout);
  std::cout << "\nOpenMP: max threads = " << omp_get_max_threads()
            << ", _OPENMP = " << _OPENMP << "\n";
  return 0;
}
