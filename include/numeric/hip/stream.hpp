#ifndef NUMERIC_HIP_STREAM_HPP_
#define NUMERIC_HIP_STREAM_HPP_

#include <hip/hip_runtime_api.h>
#include <memory>
#include <numeric/hip/device.hpp>

namespace numeric::hip {

class Stream {
public:
  Stream();
  Stream(const Device &device);
  Stream(const Stream &) = default;
  Stream(Stream &&) = default;
  Stream &operator=(const Stream &) = default;
  Stream &operator=(Stream &&) = default;

  static Stream create(const Device &device);

  hipStream_t id() const;
  Device device() const;
  bool is_running() const;
  void sync() const;

private:
  Device device_;
  std::shared_ptr<hipStream_t> id_;
};

} // namespace numeric::hip

#endif
