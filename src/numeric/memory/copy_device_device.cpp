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
  #include <numeric/math/array_op.hpp>

  template<typename Scalar, numeric::dim_t N, typename Src>
  __global__ void copy_naive(numeric::memory::ArrayView<Scalar, N> dst, Src src) {
    //copy_naive_elm(dst, src);
    dst = src;
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
