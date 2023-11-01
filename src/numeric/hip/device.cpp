#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <numeric/hip/device.hpp>
#include <numeric/hip/safe_call.hpp>
#include <numeric/math/functions.hpp>

namespace numeric::hip {

Device::Device() {
  NUMERIC_CHECK_HIP(hipGetDevice(&id_));
  // Initialize context
  NUMERIC_CHECK_HIP(hipFree(0));
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
  hipDeviceGetAttribute(&pi, hipDeviceAttributeMaxBlockDimX, id_);
  return pi;
}

unsigned Device::max_block_dim_y() const {
  int pi;
  hipDeviceGetAttribute(&pi, hipDeviceAttributeMaxBlockDimY, id_);
  return pi;
}

unsigned Device::max_block_dim_z() const {
  int pi;
  hipDeviceGetAttribute(&pi, hipDeviceAttributeMaxBlockDimZ, id_);
  return pi;
}

unsigned Device::max_grid_dim_x() const {
  int pi;
  hipDeviceGetAttribute(&pi, hipDeviceAttributeMaxGridDimX, id_);
  return pi;
}

unsigned Device::max_grid_dim_y() const {
  int pi;
  hipDeviceGetAttribute(&pi, hipDeviceAttributeMaxGridDimY, id_);
  return pi;
}

unsigned Device::max_grid_dim_z() const {
  int pi;
  hipDeviceGetAttribute(&pi, hipDeviceAttributeMaxGridDimZ, id_);
  return pi;
}

LaunchParams Device::launch_params_for_grid(unsigned Nx, unsigned Ny,
                                            unsigned Nz) const {
  LaunchParams lp;
  lp.block_dim_x = math::min(Nx, max_block_dim_x());
  lp.block_dim_y = math::min(Ny, max_block_dim_y());
  lp.block_dim_z = math::min(Nz, max_block_dim_z());
  lp.grid_dim_x = math::div_up(Nx, lp.block_dim_x);
  lp.grid_dim_y = math::div_up(Ny, lp.block_dim_y);
  lp.grid_dim_z = math::div_up(Nz, lp.block_dim_z);
  return lp;
}

} // namespace numeric::hip
