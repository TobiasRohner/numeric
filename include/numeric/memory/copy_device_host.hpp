#ifndef NUMERIC_MEMORY_COPY_DEVICE_HOST_HPP_
#define NUMERIC_MEMORY_COPY_DEVICE_HOST_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_view_decl.hpp>
#include <numeric/memory/copy_device_device.hpp>
#include <numeric/memory/copy_host_host.hpp>
#include <numeric/memory/copyer_impl.hpp>
#include <numeric/memory/device_memory_resource.hpp>
#include <numeric/memory/memcpy.hpp>
#include <numeric/memory/pinned_memory_resource.hpp>

namespace numeric::memory {

/**
 * @brief Class for copying data from device to host.
 *
 * This class provides functionality to copy data from device to host.
 *
 * @tparam Scalar The type of the elements in the arrays.
 * @tparam N The dimensionality of the arrays.
 * @tparam Src The type of the source array.
 */
template <typename Scalar, dim_t N, typename Src>
class CopyDeviceToHost : public CopyerImpl<Scalar, N, Src> {
  using super = CopyerImpl<Scalar, N, Src>;

public:
  /**
   * @brief Constructs a CopyDeviceToHost object.
   *
   * @param dst The destination array view.
   * @param src The source array.
   * @param allocate_workspace Whether to allocate workspace or not.
   */
  CopyDeviceToHost(const ArrayView<Scalar, N> &dst, const ArrayBase<Src> &src,
                   bool allocate_workspace = true)
      : allocate_workspace_(allocate_workspace) {
    NUMERIC_ERROR_IF(!allocate_workspace,
                     "Setting workspace manually is not supported yet");
    if (allocate_workspace_) {
      Layout<Src::dim> layout = src.shape();
      Scalar *ptr_device = res_device_.allocate(src.size());
      buffer_device_.set(ptr_device, layout, MemoryType::DEVICE);
      Scalar *ptr_host = res_pinned_.allocate(src.size());
      buffer_host_.set(ptr_host, layout, MemoryType::PINNED);
    }
  }

  ~CopyDeviceToHost() {
    if (allocate_workspace_) {
      if (buffer_device_.raw()) {
        res_device_.deallocate(buffer_device_.raw(), buffer_device_.size());
      }
      if (buffer_host_.raw()) {
        res_pinned_.deallocate(buffer_host_.raw(), buffer_host_.size());
      }
    }
  }

  /**
   * @brief Performs the copy operation from device to host.
   *
   * @param dst The destination array view.
   * @param src The source array.
   */
  virtual void operator()(ArrayView<Scalar, N> dst,
                          const ArrayBase<Src> &src) override {
    cpy_d_d_(buffer_device_, src);
    memcpy(buffer_host_, buffer_device_);
    cpy_h_h_(dst, buffer_host_.const_view());
  }

private:
  bool allocate_workspace_;
  CopyDeviceToDevice<Scalar, Src::dim, Src> cpy_d_d_;
  CopyHostToHost<Scalar, N, ArrayConstView<Scalar, Src::dim>> cpy_h_h_;
  DeviceMemoryResource<Scalar> res_device_;
  PinnedMemoryResource<Scalar> res_pinned_;
  ArrayView<Scalar, Src::dim> buffer_device_;
  ArrayView<Scalar, Src::dim> buffer_host_;
};

} // namespace numeric::memory

#endif
