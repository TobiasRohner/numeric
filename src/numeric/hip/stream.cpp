#include <numeric/hip/safe_call.hpp>
#include <numeric/hip/stream.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::hip {

Stream::Stream()
    : device_(), id_(new hipStream_t{0}, [](hipStream_t *id) { delete id; }) {}

Stream::Stream(const Device &device) : device_(device) {
  device.do_while_active([&]() {
    id_ = std::shared_ptr<hipStream_t>(new hipStream_t{0},
                                       [](hipStream_t *id) { delete id; });
  });
}

Stream Stream::create(const Device &device) {
  hipStream_t id;
  device.do_while_active([&]() { NUMERIC_CHECK_HIP(hipStreamCreate(&id)); });
  Stream stream;
  stream.device_ = device;
  stream.id_ =
      std::shared_ptr<hipStream_t>(new hipStream_t{id}, [](hipStream_t *id) {
        NUMERIC_CHECK_HIP(hipStreamDestroy(*id));
        delete id;
      });
  return stream;
}

hipStream_t Stream::id() const { return *id_; }

Device Stream::device() const { return device_; }

bool Stream::is_running() const {
  const hipError_t status = hipStreamQuery(*id_);
  if (status == hipSuccess) {
    return false;
  } else if (status == hipErrorNotReady) {
    return true;
  } else {
    NUMERIC_CHECK_HIP(status);
    NUMERIC_ERROR("This should never be reached");
  }
}

void Stream::sync() const { NUMERIC_CHECK_HIP(hipStreamSynchronize(*id_)); }

} // namespace numeric::hip
