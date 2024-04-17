#ifndef NUMERIC_MEMORY_COPY_DEVICE_DEVICE_HPP_
#define NUMERIC_MEMORY_COPY_DEVICE_DEVICE_HPP_

#include <numeric/config.hpp>
#include <numeric/hip/device.hpp>
#include <numeric/hip/kernel.hpp>
#include <numeric/hip/launch_params.hpp>
#include <numeric/memory/array_view_decl.hpp>
#include <numeric/memory/copyer_impl.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/utils/type_name.hpp>

namespace numeric::memory {

namespace internal {

/**
 * @brief Builds the kernel for copying data from device to device.
 *
 * This function constructs the kernel for copying data from device to device.
 *
 * @param scalar The type name of the scalar of the destination array.
 * @param N The dimensionality of the data.
 * @param src The type name of the source array.
 * @return hip::Kernel The constructed kernel.
 */
hip::Kernel copy_device_to_device_build_kernel(std::string_view scalar, dim_t N,
                                               std::string_view src);

} // namespace internal

/**
 * @brief Class for copying data between two arrays on a device.
 *
 * This class provides functionality to copy data between two arrays on a
 * device.
 *
 * @tparam Scalar The type of the elements in the destination array.
 * @tparam N The dimensionality of the destination array.
 * @tparam Src The type of the source array.
 */
template <typename Scalar, dim_t N, typename Src>
class CopyDeviceToDevice : public CopyerImpl<Scalar, N, Src> {
  using super = CopyerImpl<Scalar, N, Src>;

public:
  /**
   * @brief Constructs a CopyDeviceToDevice object.
   *
   * @param device The device to use for the copy operation.
   */
  CopyDeviceToDevice(const hip::Device &device = hip::Device())
      : device_(device) {
    static hip::Kernel shared_kernel =
        internal::copy_device_to_device_build_kernel(
            utils::type_name<Scalar>(), N,
            utils::type_name<decltype(meta::declval<Src>().broadcast(
                meta::declval<ArrayView<Scalar, N>>().shape()))>());
    kernel_ = shared_kernel;
  }

  /**
   * @brief Performs the copy operation from device to device.
   *
   * @param dst The destination array view.
   * @param src The source array.
   */
  virtual void operator()(ArrayView<Scalar, N> dst,
                          const ArrayBase<Src> &src) override {
    const auto src_derived = src.derived().broadcast(dst.shape());
    hip::LaunchParams lp;
    if (N == 1) {
      lp = device_.launch_params_for_grid(dst.shape(0), 1, 1);
    } else if (N == 2) {
      lp = device_.launch_params_for_grid(dst.shape(1), dst.shape(0), 1);
    } else {
      lp = device_.launch_params_for_grid(dst.shape(2), dst.shape(1),
                                          dst.shape(0));
    }
    kernel_(lp, hip::Stream(device_), dst, src_derived);
  }

private:
  hip::Device device_;
  hip::Kernel kernel_;
};

} // namespace numeric::memory

#endif
