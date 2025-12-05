#include <numeric/equations/fem/finite_element_vector.hpp>
#include <string>

namespace numeric::equations::fem {

namespace internal {

static const char kernel_includes[] = R"(
  #include <numeric/memory/array_const_view.hpp>
  #include <numeric/memory/array_view.hpp>
  #include <numeric/math/basis_lagrange.hpp>
  #include <numeric/math/fes/basis_h1.hpp>
  #include <numeric/math/fes/basis_l2.hpp>
  #include <numeric/equations/fem/load_element_vector.hpp>
)";

static const char kernel_src[] = R"(
  template <typename Scalar, typename ScalarMesh, typename ElementVector>
  __global__ void build_vector(
      ElementVector element_vector,
      numeric::memory::ArrayConstView<numeric::dim_t, 1> group,
      numeric::memory::ArrayConstView<ScalarMesh, 2> vertices,
      numeric::memory::ArrayConstView<numeric::dim_t, 2> elements,
      numeric::memory::ArrayConstView<numeric::dim_t, 2> dofs,
      numeric::memory::ArrayView<Scalar, 1> out) {
    static constexpr numeric::dim_t num_nodes = ElementVector::element_t::num_nodes;
    static constexpr numeric::dim_t num_basis_functions = ElementVector::num_basis_functions;
    const numeric::dim_t num_elements = group.shape(0);
    const numeric::dim_t world_dim = vertices.shape(0);

    const numeric::dim_t tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (tid >= num_elements) {
      return;
    }

    extern __shared__ Scalar work[];
    Scalar *local_work =
	work + hipThreadIdx_x * (world_dim * num_nodes + ElementVector::apply_work_size(world_dim) / sizeof(Scalar));
    Scalar (*nodes)[num_nodes] = reinterpret_cast<Scalar(*)[num_nodes]>(local_work);
    Scalar *apply_work = local_work + world_dim * num_nodes;
    Scalar local_vector[num_basis_functions];

    const numeric::dim_t element = group(tid);

    // Extract physical coordinates of the current element's nodes
    for (numeric::dim_t node = 0 ; node < num_nodes ; ++node) {
      for (numeric::dim_t dim = 0 ; dim < world_dim ; ++dim) {
	nodes[dim][node] = vertices(dim, elements(node, element));
      }
    }

    // Apply the element-local operator
    element_vector.apply(f, nodes, world_dim, local_vector, apply_work);

    // Store the local vector to global memory for the gather operation
    for (numeric::dim_t bf = 0 ; bf < num_basis_functions ; ++bf) {
      out(dofs(bf, element)) += local_vector[bf];
    }
  }
)";

hip::Kernel element_vector_build_kernel(std::string_view scalar,
                                        std::string_view scalar_mesh,
                                        std::string_view element_vector,
                                        std::string_view f) {
  const std::string kernel_name = "build_vector<" + std::string(scalar) + ", " +
                                  std::string(scalar_mesh) + ", " +
                                  std::string(element_vector) + ">";
  hip::Program program(std::string(kernel_includes) + "\nstatic auto f = " +
                       std::string(f) + ";\n" + std::string(kernel_src));
  program.add_compile_option("--device-as-default-execution-space");
  program.instantiate_kernel(kernel_name);
  return program.get_kernel(kernel_name);
}

} // namespace internal

} // namespace numeric::equations::fem
