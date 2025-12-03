#include <numeric/hip/device.hpp>
#include <numeric/hip/runtime.hpp>
#include <numeric/hip/safe_call.hpp>
#include <numeric/math/functions.hpp>

namespace numeric::hip {

Device::Device() {
  // Initialize context
  NUMERIC_CHECK_HIP(hipFree(0));
  NUMERIC_CHECK_HIP(hipGetDevice(&id_));
}

Device::Device(int id) : id_(id) {
  // Initialize context
  int current_id;
  NUMERIC_CHECK_HIP(hipGetDevice(&current_id));
  NUMERIC_CHECK_HIP(hipFree(0));
  NUMERIC_CHECK_HIP(hipSetDevice(current_id));
}

int Device::count() {
  int c;
  NUMERIC_CHECK_HIP(hipGetDeviceCount(&c));
  return c;
}

void Device::activate() const { NUMERIC_CHECK_HIP(hipSetDevice(id_)); }

void Device::sync() const {
  do_while_active([&]() { NUMERIC_CHECK_HIP(hipDeviceSynchronize()); });
}

int Device::id() const { return id_; }

unsigned Device::max_block_dim_x() const {
  int pi;
  NUMERIC_CHECK_HIP(
      hipDeviceGetAttribute(&pi, hipDeviceAttributeMaxBlockDimX, id_));
  return pi;
}

unsigned Device::max_block_dim_y() const {
  int pi;
  NUMERIC_CHECK_HIP(
      hipDeviceGetAttribute(&pi, hipDeviceAttributeMaxBlockDimY, id_));
  return pi;
}

unsigned Device::max_block_dim_z() const {
  int pi;
  NUMERIC_CHECK_HIP(
      hipDeviceGetAttribute(&pi, hipDeviceAttributeMaxBlockDimZ, id_));
  return pi;
}

unsigned Device::max_grid_dim_x() const {
  int pi;
  NUMERIC_CHECK_HIP(
      hipDeviceGetAttribute(&pi, hipDeviceAttributeMaxGridDimX, id_));
  return pi;
}

unsigned Device::max_grid_dim_y() const {
  int pi;
  NUMERIC_CHECK_HIP(
      hipDeviceGetAttribute(&pi, hipDeviceAttributeMaxGridDimY, id_));
  return pi;
}

unsigned Device::max_grid_dim_z() const {
  int pi;
  NUMERIC_CHECK_HIP(
      hipDeviceGetAttribute(&pi, hipDeviceAttributeMaxGridDimZ, id_));
  return pi;
}

unsigned Device::max_threads_per_block() const {
  int pi;
  NUMERIC_CHECK_HIP(
      hipDeviceGetAttribute(&pi, hipDeviceAttributeMaxThreadsPerBlock, id_));
  return pi;
}

unsigned Device::max_shared_memory_per_block() const {
  int pi;
  NUMERIC_CHECK_HIP(hipDeviceGetAttribute(
      &pi, hipDeviceAttributeMaxSharedMemoryPerBlock, id_));
  return pi;
}

int Device::warp_size() const {
  hipDeviceProp_t props;
  NUMERIC_CHECK_HIP(hipGetDeviceProperties(&props, id_));
  return props.warpSize;
}

LaunchParams Device::launch_params_for_grid(unsigned Nx, unsigned Ny,
                                            unsigned Nz) const {
  const unsigned max_threads = max_threads_per_block();
  LaunchParams lp;
  lp.block_dim_x = math::min(Nx, max_block_dim_x());
  lp.block_dim_y =
      math::min(math::min(Ny, max_block_dim_y()), max_threads / lp.block_dim_x);
  lp.block_dim_z = math::min(math::min(Nz, max_block_dim_z()),
                             max_threads / (lp.block_dim_x * lp.block_dim_y));
  lp.grid_dim_x = math::div_up(Nx, lp.block_dim_x);
  lp.grid_dim_y = math::div_up(Ny, lp.block_dim_y);
  lp.grid_dim_z = math::div_up(Nz, lp.block_dim_z);
  lp.shared_mem_bytes = 0;
  return lp;
}

} // namespace numeric::hip
