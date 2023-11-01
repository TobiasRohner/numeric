#ifndef NUMERIC_MEMORY_COPYER_HPP_
#define NUMERIC_MEMORY_COPYER_HPP_

#include <memory>
#include <numeric/config.hpp>
#include <numeric/memory/array_view_decl.hpp>
#include <numeric/memory/copy_host_host.hpp>
#include <numeric/memory/copyer_impl.hpp>
#if NUMERIC_ENABLE_HIP
#include <numeric/memory/copy_device_device.hpp>
#include <numeric/memory/copy_device_host.hpp>
#include <numeric/memory/copy_host_device.hpp>
#endif

namespace numeric::memory {

template <typename Scalar, dim_t N, typename Src> class Copyer {
  using impl_t = CopyerImpl<Scalar, N, Src>;

public:
  Copyer(const std::shared_ptr<impl_t> &impl) : impl_(impl) {}
  Copyer(ArrayView<Scalar, N> dst, const Src &src)
      : Copyer(make_copyer_impl(dst, src)) {}
  Copyer(const Copyer &) = default;
  Copyer(Copyer &&) = default;
  ~Copyer() = default;
  Copyer &operator=(const Copyer &) = default;
  Copyer &operator=(Copyer &&) = default;

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

} // namespace numeric::memory

#endif
