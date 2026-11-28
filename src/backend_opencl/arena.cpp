#include "arena.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "bench/datagen.hpp"
#include "bench/init_check.hpp"
#include "backend_opencl/program.hpp"
#include "cl_sources/init.cl.hpp"

namespace bench {

namespace {

constexpr std::size_t init_check_count = 32;  // random samples per matrix (plus corners)

// Sub-buffer [offset, offset + bytes) of parent
ClMem make_sub_buffer(const ClMem& parent, std::size_t offset, std::size_t bytes) {
  const cl_buffer_region region{offset, bytes};
  cl_int err = CL_SUCCESS;
  ClMem sub(clCreateSubBuffer(parent.get(), CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err));
  cl_check(err, "clCreateSubBuffer");
  return sub;
}

// Address of buffer as seen by the kernels (base_address kernel, one work-item)
std::uintptr_t kernel_address(const ClRuntime& rt, const ClProgram& program, const ClMem& buffer) {
  cl_int err = CL_SUCCESS;
  const ClMem out(clCreateBuffer(rt.context.get(), CL_MEM_WRITE_ONLY, sizeof(cl_ulong), nullptr, &err));
  cl_check(err, "clCreateBuffer");
  const ClKernel kernel = make_kernel(program, "base_address");
  set_args(kernel, buffer.get(), out.get());
  enqueue_1d(rt, kernel, 1);
  cl_ulong address = 0;
  cl_check(clEnqueueReadBuffer(rt.queue.get(), out.get(), CL_TRUE, 0, sizeof(address), &address, 0, nullptr,
                               nullptr),
           "clEnqueueReadBuffer");
  return static_cast<std::uintptr_t>(address);
}

// rows x ld matrix: value function for col < cols, NaN in the row padding
void fill_matrix(const ClRuntime& rt, const ClProgram& program, const ClMem& buf, std::size_t rows,
                 std::size_t cols, std::size_t ld, MatrixId id, std::uint64_t seed) {
  const ClKernel kernel = make_kernel(program, "fill_matrix");
  set_args(kernel, buf.get(), cl_ulong{cols}, cl_ulong{ld}, static_cast<cl_ulong>(id), cl_ulong{seed});
  // One work-item per element, including the padding columns; dimension 0 = column
  enqueue_2d(rt, kernel, ld, rows);
}

// count consecutive doubles set to NaN
void fill_nan(const ClRuntime& rt, const ClProgram& program, const ClMem& buf, std::size_t count) {
  const ClKernel kernel = make_kernel(program, "fill_nan");
  set_args(kernel, buf.get());
  enqueue_1d(rt, kernel, count);
}

} // namespace

Arena::Arena(const ClRuntime& rt, const Problem& p, std::size_t workspace_bytes, bool require_alignment)
    : rt_(rt), p_(p) {
  if (workspace_bytes % sizeof(double) != 0)
    throw std::invalid_argument("workspace size must be a multiple of sizeof(double)");

  // Offsets and sizes of A, B, C, workspace relative to the arena base
  const ArenaLayout layout = compute_layout(p, workspace_bytes);
  info_.alloc_kind = "clCreateBuffer";
  info_.layout = layout;

  // Sub-buffer offsets must be multiples of CL_DEVICE_MEM_BASE_ADDR_ALIGN (in bits)
  cl_uint align_bits = 0;
  cl_check(clGetDeviceInfo(rt_.device.device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(align_bits), &align_bits,
                           nullptr),
           "clGetDeviceInfo");
  if (align_bits == 0 || (arena_page * 8) % align_bits != 0)
    throw std::runtime_error("CL_DEVICE_MEM_BASE_ADDR_ALIGN = " + std::to_string(align_bits) +
                             " bits does not divide the region offsets (multiples of " +
                             std::to_string(arena_page) + " bytes)");

  // Single buffer for all regions, allocated by the implementation
  cl_int err = CL_SUCCESS;
  base_ = ClMem(clCreateBuffer(rt_.context.get(), CL_MEM_READ_WRITE, layout.total_bytes, nullptr, &err));
  cl_check(err, ("clCreateBuffer of " + std::to_string(layout.total_bytes) + " bytes").c_str());

  // Init kernels, compiled once here (outside the timed iterations)
  init_program_ = build_program(rt_, cl_source::init, "");

  // Record the base address the kernels see; abort only if required
  record_alignment(info_, reinterpret_cast<const void*>(kernel_address(rt_, init_program_, base_)),
                   require_alignment);
  info_.requested_alignment = 0;  // no alignment can be requested from clCreateBuffer

  m_.a = make_sub_buffer(base_, layout.a_offset, layout.a_bytes);
  m_.b = make_sub_buffer(base_, layout.b_offset, layout.b_bytes);
  m_.c = make_sub_buffer(base_, layout.c_offset, layout.c_bytes);
  if (workspace_bytes > 0) m_.workspace = make_sub_buffer(base_, layout.ws_offset, workspace_bytes);
  m_.workspace_bytes = workspace_bytes;
}

void Arena::initialize(std::uint64_t seed) {
  // Independent fill kernels (in-order queue), then wait for all of them
  fill_matrix(rt_, init_program_, m_.a, p_.m, p_.k, p_.lda, MatrixId::A, seed);
  fill_matrix(rt_, init_program_, m_.b, p_.k, p_.n, p_.ldb, MatrixId::B, seed);
  fill_nan(rt_, init_program_, m_.c, p_.m * p_.ldc);
  if (m_.workspace_bytes > 0)
    fill_nan(rt_, init_program_, m_.workspace, m_.workspace_bytes / sizeof(double));
  cl_check(clFinish(rt_.queue.get()), "clFinish");

  // Read back sampled elements and compare with the host value function (one read per sample)
  const auto samples = make_init_samples(p_, seed, init_check_count);
  std::vector<double> values(samples.size());
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const auto& s = samples[i];
    const ClMem& src = s.matrix == InitSample::Matrix::A ? m_.a
                     : s.matrix == InitSample::Matrix::B ? m_.b
                                                         : m_.c;
    cl_check(clEnqueueReadBuffer(rt_.queue.get(), src.get(), CL_FALSE, s.offset * sizeof(double), sizeof(double),
                                 &values[i], 0, nullptr, nullptr),
             "clEnqueueReadBuffer");
  }
  cl_check(clFinish(rt_.queue.get()), "clFinish");  // all reads done before the values are checked
  check_init_samples(p_, seed, samples, values);
}

void Arena::copy_c_to_host(double* dst) const {
  // Whole C including the row padding (M x ldc elements); blocking read
  cl_check(clEnqueueReadBuffer(rt_.queue.get(), m_.c.get(), CL_TRUE, 0, info_.layout.c_bytes, dst, 0, nullptr,
                               nullptr),
           "clEnqueueReadBuffer");
}

} // namespace bench
