#ifndef NUMERIC_MEMORY_COPYER_HPP_
#define NUMERIC_MEMORY_COPYER_HPP_

#include <memory>
#include <numeric/config.hpp>
#include <numeric/memory/array_view_decl.hpp>
#include <numeric/memory/copy_host_host.hpp>
#include <numeric/memory/copyer_impl.hpp>
#include <numeric/utils/error.hpp>
#if NUMERIC_ENABLE_HIP
#include <numeric/memory/copy_device_device.hpp>
#include <numeric/memory/copy_device_host.hpp>
#include <numeric/memory/copy_host_device.hpp>
#endif

namespace numeric::memory {

/**
 * @brief Class for copying data between different memory locations.
 *
 * This class provides functionality to copy data between memory locations,
 * such as host to host, host to device, device to host, and device to device.
 *
 * @tparam Scalar The type of the elements in the destination array.
 * @tparam N The dimensionality of the destination array.
 * @tparam Src The type of the source array.
 */
template <typename Scalar, dim_t N, typename Src> class Copyer {
  using impl_t = CopyerImpl<Scalar, N, Src>;

public:
  /**
   * @brief Constructs a Copyer object with a specified implementation.
   *
   * @param impl The implementation of the copy operation.
   */
  Copyer(const std::shared_ptr<impl_t> &impl) : impl_(impl) {}

  /**
   * @brief Constructs a Copyer object with destination and source arrays.
   *
   * @param dst The destination array view.
   * @param src The source array.
   */
  Copyer(ArrayView<Scalar, N> dst, const Src &src)
      : Copyer(make_copyer_impl(dst, src)) {}

  Copyer(const Copyer &) = default;
  Copyer(Copyer &&) = default;
  ~Copyer() = default;
  Copyer &operator=(const Copyer &) = default;
  Copyer &operator=(Copyer &&) = default;

  /**
   * @brief Performs the copy operation.
   *
   * @param dst The destination array view.
   * @param src The source array.
   */
  void operator()(ArrayView<Scalar, N> dst, const ArrayBase<Src> &src) {
    impl_->operator()(dst, src);
  }

private:
  std::shared_ptr<impl_t> impl_;

  static std::shared_ptr<impl_t> make_copyer_impl(ArrayView<Scalar, N> dst,
                                                  const Src &src) {
    if (is_host_accessible(dst.memory_type()) &&
        is_host_accessible(src.memory_type())) {
      return std::make_shared<CopyHostToHost<Scalar, N, Src>>();
#if NUMERIC_ENABLE_HIP
    } else if (is_device_accessible(dst.memory_type()) &&
               is_host_accessible(src.memory_type())) {
      return std::make_shared<CopyHostToDevice<Scalar, N, Src>>(dst,
                                                                src.derived());
    } else if (is_host_accessible(dst.memory_type()) &&
               is_device_accessible(src.memory_type())) {
      return std::make_shared<CopyDeviceToHost<Scalar, N, Src>>(dst,
                                                                src.derived());
    } else if (is_device_accessible(dst.memory_type()) &&
               is_device_accessible(src.memory_type())) {
      return std::make_shared<CopyDeviceToDevice<Scalar, N, Src>>();
#endif
    } else {
      NUMERIC_ERROR("Unsupported memory locations {} -> {}",
                    to_string(src.memory_type()), to_string(dst.memory_type()));
    }
  }
};

/**
 * @brief Helper function to create a Copyer object.
 *
 * @tparam Scalar The type of the elements in the arrays.
 * @tparam N The dimensionality of the arrays.
 * @tparam Src The type of the source array.
 * @param dst The destination array view.
 * @param src The source array.
 * @return Copyer<Scalar, N, Src> The created Copyer object.
 */
template <typename Scalar, dim_t N, typename Src>
Copyer<Scalar, N, Src> make_copyer(ArrayView<Scalar, N> dst, const Src &src) {
  return Copyer<Scalar, N, Src>(dst, src);
}

} // namespace numeric::memory

#endif
