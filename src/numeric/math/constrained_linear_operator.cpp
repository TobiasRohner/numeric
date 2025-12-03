#include <numeric/hip/program.hpp>
#include <numeric/math/constrained_linear_operator.hpp>

namespace numeric::math::internal {

static const char kernel_clear_constrained_src[] = R"(
  #include <numeric/memory/array_const_view.hpp>
  #include <numeric/memory/array_view.hpp>

  template <typename Scalar>
  __global__ void clear_constrained(
      numeric::memory::ArrayConstView<numeric::dim_t, 1> constrained_dofs,
      numeric::memory::ArrayView<Scalar, 1> x) {
    const numeric::dim_t tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (tid >= constrained_dofs.shape(0)) {
      return;
    }
    const numeric::dim_t dof = constrained_dofs(tid);
    x(dof) = 0;
  }
)";

static const char kernel_copy_constrained_src[] = R"(
  #include <numeric/memory/array_const_view.hpp>
  #include <numeric/memory/array_view.hpp>

  template <typename Scalar>
  __global__ void copy_constrained(
      numeric::memory::ArrayConstView<numeric::dim_t, 1> constrained_dofs,
      numeric::memory::ArrayView<Scalar, 1> x,
      numeric::memory::ArrayConstView<Scalar, 1> vals) {
    const numeric::dim_t tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (tid >= constrained_dofs.shape(0)) {
      return;
    }
    const numeric::dim_t dof = constrained_dofs(tid);
    x(dof) = vals(dof);
  }
)";

hip::Kernel build_kernel_clear_constrained_impl(std::string_view scalar) {
  const std::string kernel_name =
      "clear_constrained<" + std::string(scalar) + ">";
  hip::Program program(kernel_clear_constrained_src);
  program.instantiate_kernel(kernel_name);
  return program.get_kernel(kernel_name);
}

hip::Kernel build_kernel_copy_constrained_impl(std::string_view scalar) {
  const std::string kernel_name =
      "copy_constrained<" + std::string(scalar) + ">";
  hip::Program program(kernel_copy_constrained_src);
  program.instantiate_kernel(kernel_name);
  return program.get_kernel(kernel_name);
}

} // namespace numeric::math::internal
