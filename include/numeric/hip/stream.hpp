#ifndef NUMERIC_HIP_STREAM_HPP_
#define NUMERIC_HIP_STREAM_HPP_

#include <hip/hip_runtime_api.h>
#include <memory>
#include <numeric/hip/device.hpp>

namespace numeric::hip {

/**
 * @brief Class representing a HIP stream.
 */
class Stream {
public:
  /**
   * @brief Default constructor.
   *
   * Constructs the default stream associated with the default device.
   */
  Stream();

  /**
   * @brief Constructor with specified device.
   *
   * Constructs the default stream associated with the specified device.
   *
   * @param device Device associated with the stream.
   */
  Stream(const Device &device);

  Stream(const Stream &) = default;
  Stream(Stream &&) = default;
  Stream &operator=(const Stream &) = default;
  Stream &operator=(Stream &&) = default;

  /**
   * @brief Creates a stream associated with the specified device.
   *
   * @param device Device associated with the stream.
   * @return Stream object.
   */
  static Stream create(const Device &device);

  hipStream_t id() const;
  Device device() const;

  /**
   * @brief Checks if the stream is currently running.
   *
   * @return True if the stream is running, false otherwise.
   */
  bool is_running() const;

  /**
   * @brief Wait until all operations on the current stream have finished.
   */
  void sync() const;

private:
  Device device_;
  std::shared_ptr<hipStream_t> id_;
};

} // namespace numeric::hip

#endif
