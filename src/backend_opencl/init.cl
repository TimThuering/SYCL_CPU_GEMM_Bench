// Arena kernels of the OpenCL reference: initialization (counterpart of the init kernels in
// backend_sycl/arena.cpp) and the arena base address.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Value function, identical to include/bench/datagen.hpp (same integer operations, so the values
// are bit-identical to the host's; checked by the init sample check)
ulong splitmix64(ulong x) {
  x += 0x9E3779B97F4A7C15UL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
  return x ^ (x >> 31);
}

double gen_value(ulong seed, ulong id, ulong row, ulong col) {
  ulong h = splitmix64(seed ^ (id << 56));
  h = splitmix64(h ^ row);
  h = splitmix64(h ^ col);
  const double u = (double)(h >> 11) * 0x1.0p-53;
  return 2.0 * u - 1.0;
}

// rows x ld matrix: value function for col < cols, NaN in the row padding.
// One work-item per element, global range {ld, rows}: dimension 0 = column (fastest).
__kernel void fill_matrix(__global double* ptr, ulong cols, ulong ld, ulong id, ulong seed) {
  const ulong j = get_global_id(0);  // column
  const ulong i = get_global_id(1);  // row
  ptr[i * ld + j] = (j < cols) ? gen_value(seed, id, i, j) : (double)NAN;
}

// count consecutive doubles set to NaN, global range {count}
__kernel void fill_nan(__global double* ptr) {
  ptr[get_global_id(0)] = (double)NAN;
}

// Address of the buffer as seen by the kernels (for the alignment record), single work-item
__kernel void base_address(__global const double* base, __global ulong* out) {
  out[0] = (ulong)base;
}
