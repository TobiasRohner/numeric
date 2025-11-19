#include <numeric/equations/fem/finite_element_vector.hpp>
#include <string>

namespace numeric::equations::fem {

namespace internal {

static const char kernel_src[] = R"(
  #include <numeric/memory/array_const_view.hpp>
  #include <numeric/memory/array_view.hpp>
  #include <numeric/math/basis_lagrange.hpp>
  #include <numeric/math/fes/basis_h1.hpp>
  #include <numeric/math/fes/basis_l2.hpp>
  #include <numeric/equations/fem/load_element_vector.hpp>

  template <typename Scalar, typename ElementVector>
  __global__ void compute_element_vectors() {
    // TODO: Implement
  }

  template <typename Scalar>
  __global__ void gather_element_vectors() {
    // TODO: Implement
  }
)";

utils::Tuple<hip::Kernel, hip::Kernel>
element_vector_build_kernel(std::string_view scalar,
                            std::string_view element_vector) {
  const std::string compute_kernel_name = "compute_element_vectors<" +
                                          std::string(scalar) + ", " +
                                          std::string(element_vector) + ">";
  const std::string gather_kernel_name =
      "gather_element_vectors<" + std::string(scalar) + ">";
  hip::Program program(kernel_src);
  program.instantiate_kernel(compute_kernel_name);
  program.instantiate_kernel(gather_kernel_name);
  auto compute_kernel = program.get_kernel(compute_kernel_name);
  auto gather_kernel = program.get_kernel(gather_kernel_name);
  return utils::Tuple<hip::Kernel, hip::Kernel>(std::move(compute_kernel),
                                                std::move(gather_kernel));
}

} // namespace internal

} // namespace numeric::equations::fem
