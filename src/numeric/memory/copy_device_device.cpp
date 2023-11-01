#include <numeric/hip/program.hpp>
#include <numeric/memory/copy_device_device.hpp>
#include <string>

namespace numeric::memory {

namespace internal {

static const char kernel_src[] = R"(
  #include <numeric/memory/copy_kernels.hpp>
  #include <numeric/memory/array_base.hpp>
  #include <numeric/memory/array_const_view.hpp>
  #include <numeric/memory/array_view.hpp>
  #include <numeric/memory/array_op.hpp>

  template<typename Scalar, typename Src>
  __device__ void copy_naive_impl(numeric::memory::ArrayView<Scalar, 1> &dst, const Src &src) {
    const numeric::dim_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i >= dst.shape(0)) {
      return;
    }
    dst(i) = src(i);
  }

  template<typename Scalar, typename Src>
  __device__ void copy_naive_impl(numeric::memory::ArrayView<Scalar, 2> &dst, const Src &src) {
    const numeric::dim_t i = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const numeric::dim_t j = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i >= dst.shape(0)) {
      return;
    }
    if (j >= dst.shape(1)) {
      return;
    }
    dst(i, j) = src(i, j);
  }

  template<typename Scalar, numeric::dim_t N, typename Src>
  __device__ void copy_naive_impl(numeric::memory::ArrayView<Scalar, N> &dst, const Src &src) {
    const numeric::dim_t i = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    const numeric::dim_t j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    const numeric::dim_t k = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i >= dst.shape(0)) {
      return;
    }
    if (j >= dst.shape(1)) {
      return;
    }
    if (k >= dst.shape(2)) {
      return;
    }
    dst(i, j, k) = src(i, j, k);
  }

  template<typename Scalar, numeric::dim_t N, typename Src>
  __global__ void copy_naive(numeric::memory::ArrayView<Scalar, N> dst, Src src) {
    copy_naive_impl(dst, src);
  }
)";

hip::Kernel copy_device_to_device_build_kernel(std::string_view scalar, dim_t N,
                                               std::string_view src) {
  const std::string kernel_name = "copy_naive<" + std::string(scalar) + ", " +
                                  std::to_string(N) + ", " + std::string(src) +
                                  ">";
  hip::Program program(kernel_src);
  program.instantiate_kernel(kernel_name);
  return program.get_kernel(kernel_name);
}

} // namespace internal

} // namespace numeric::memory
