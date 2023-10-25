#ifndef NUMERIC_MEMORY_COPY_DEVICE_DEVICE_HPP_
#define NUMERIC_MEMORY_COPY_DEVICE_DEVICE_HPP_

#include <numeric/config.hpp>
#include <numeric/hip/kernel.hpp>
#include <numeric/memory/array_view_decl.hpp>
#include <numeric/utils/type_name.hpp>

namespace numeric::memory {

namespace internal {

hip::Kernel copy_device_to_device_build_kernel(std::string_view scalar, dim_t N,
                                               std::string_view src);

}

template <typename Scalar, dim_t N, typename Src> class CopyDeviceToDevice {
public:
  CopyDeviceToDevice()
      : kernel_(internal::copy_device_to_device_build_kernel(
            utils::type_name<Scalar>(), N, utils::type_name<Src>())) {}

  void operator()(ArrayView<Scalar, N> dst, const ArrayBase<Src> &src) const {
    const Src src_derived = src.derived();
    kernel_(1, 1, 1, 1, 1, 1, 0, hip::Stream(), dst, src_derived);
  }

private:
  hip::Kernel kernel_;
};

} // namespace numeric::memory

#endif
