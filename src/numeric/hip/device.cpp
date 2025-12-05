#include <numeric/hip/device.hpp>
#include <numeric/hip/runtime.hpp>
#include <numeric/hip/safe_call.hpp>
#include <numeric/math/functions.hpp>

namespace numeric::hip {

std::map<int, hipDeviceProp_t> Device::props_ =
    std::map<int, hipDeviceProp_t>();

Device::Device() {
  // Initialize context
  NUMERIC_CHECK_HIP(hipFree(0));
  NUMERIC_CHECK_HIP(hipGetDevice(&id_));
  auto it = props_.find(id_);
  if (it == props_.end()) {
    hipDeviceProp_t p;
    NUMERIC_CHECK_HIP(hipGetDeviceProperties(&p, id_));
    props_[id_] = p;
  }
}

Device::Device(int id) : id_(id) {
  // Initialize context
  int current_id;
  NUMERIC_CHECK_HIP(hipGetDevice(&current_id));
  NUMERIC_CHECK_HIP(hipFree(0));
  NUMERIC_CHECK_HIP(hipSetDevice(current_id));
  auto it = props_.find(id_);
  if (it == props_.end()) {
    hipDeviceProp_t p;
    NUMERIC_CHECK_HIP(hipGetDeviceProperties(&p, id_));
    props_[id_] = p;
  }
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

int Device::max_block_dim_x() const { return props_[id_].maxThreadsDim[0]; }

int Device::max_block_dim_y() const { return props_[id_].maxThreadsDim[1]; }

int Device::max_block_dim_z() const { return props_[id_].maxThreadsDim[2]; }

int Device::max_grid_dim_x() const { return props_[id_].maxGridSize[0]; }

int Device::max_grid_dim_y() const { return props_[id_].maxGridSize[1]; }

int Device::max_grid_dim_z() const { return props_[id_].maxGridSize[2]; }

int Device::max_threads_per_block() const {
  return props_[id_].maxThreadsPerBlock;
}

size_t Device::max_shared_memory_per_block() const {
  return props_[id_].sharedMemPerBlock;
}

size_t Device::reserved_shared_memory_per_block() const {
  return props_[id_].reservedSharedMemPerBlock;
}

size_t Device::available_shared_memory_per_block() const {
  return max_shared_memory_per_block() - reserved_shared_memory_per_block();
}

int Device::warp_size() const { return props_[id_].warpSize; }

LaunchParams Device::launch_params_for_grid(unsigned Nx, unsigned Ny,
                                            unsigned Nz) const {
  const unsigned max_threads = max_threads_per_block();
  LaunchParams lp;
  lp.block_dim_x = math::min(Nx, static_cast<unsigned>(max_block_dim_x()));
  lp.block_dim_y =
      math::min(math::min(Ny, static_cast<unsigned>(max_block_dim_y())),
                max_threads / lp.block_dim_x);
  lp.block_dim_z =
      math::min(math::min(Nz, static_cast<unsigned>(max_block_dim_z())),
                max_threads / (lp.block_dim_x * lp.block_dim_y));
  lp.grid_dim_x = math::div_up(Nx, lp.block_dim_x);
  lp.grid_dim_y = math::div_up(Ny, lp.block_dim_y);
  lp.grid_dim_z = math::div_up(Nz, lp.block_dim_z);
  lp.shared_mem_bytes = 0;
  return lp;
}

} // namespace numeric::hip
