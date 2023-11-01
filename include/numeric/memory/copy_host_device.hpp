#ifndef NUMERIC_MEMORY_COPY_HOST_DEVICE_HPP_
#define NUMERIC_MEMORY_COPY_HOST_DEVICE_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_view_decl.hpp>
#include <numeric/memory/copy_device_device.hpp>
#include <numeric/memory/copy_host_host.hpp>
#include <numeric/memory/copyer_impl.hpp>
#include <numeric/memory/device_memory_resource.hpp>
#include <numeric/memory/memcpy.hpp>
#include <numeric/memory/pinned_memory_resource.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::memory {

template <typename Scalar, dim_t N, typename Src>
class CopyHostToDevice : public CopyerImpl<Scalar, N, Src> {
  using super = CopyerImpl<Scalar, N, Src>;

public:
  CopyHostToDevice(const ArrayView<Scalar, N> &dst, const ArrayBase<Src> &src,
                   bool allocate_workspace = true)
      : allocate_workspace_(allocate_workspace) {
    NUMERIC_ERROR_IF(!allocate_workspace,
                     "Setting workspace manually is not supported yet");
    if (allocate_workspace_) {
      Layout<Src::dim> layout = src.layout();
      layout.stride(Src::dim - 1) = 1;
      for (dim_t d = Src::dim - 2; d >= 0; --d) {
        layout.stride(d) = layout.stride(d + 1) * layout.shape(d + 1);
      }
      Scalar *ptr_host = res_pinned_.allocate(src.size());
      buffer_host_.set(ptr_host, layout, MemoryType::PINNED);
      Scalar *ptr_device = res_device_.allocate(src.size());
      buffer_device_.set(ptr_device, layout, MemoryType::DEVICE);
    }
  }

  ~CopyHostToDevice() {
    if (allocate_workspace_) {
      if (buffer_host_.raw()) {
        res_pinned_.deallocate(buffer_host_.raw(), buffer_host_.size());
      }
      if (buffer_device_.raw()) {
        res_device_.deallocate(buffer_device_.raw(), buffer_device_.size());
      }
    }
  }

  virtual void operator()(ArrayView<Scalar, N> dst,
                          const ArrayBase<Src> &src) override {
    cpy_h_h_(buffer_host_, src);
    memcpy(buffer_device_, buffer_host_);
    cpy_d_d_(dst, buffer_device_.const_view());
  }

private:
  bool allocate_workspace_;
  CopyHostToHost<Scalar, Src::dim, Src> cpy_h_h_;
  CopyDeviceToDevice<Scalar, N, ArrayConstView<Scalar, Src::dim>> cpy_d_d_;
  PinnedMemoryResource<Scalar> res_pinned_;
  DeviceMemoryResource<Scalar> res_device_;
  ArrayView<Scalar, Src::dim> buffer_host_;
  ArrayView<Scalar, Src::dim> buffer_device_;
};

} // namespace numeric::memory

#endif
