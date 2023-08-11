#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <numeric/hip/device.hpp>
#include <numeric/hip/safe_call.hpp>

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

} // namespace numeric::hip
