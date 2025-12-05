#include <numeric/equations/fem/finite_element_matrix.hpp>
#include <string>

namespace numeric::equations::fem {

namespace internal {

static const char kernel_src[] = R"(
  #include <numeric/memory/array_const_view.hpp>
  #include <numeric/memory/array_view.hpp>
  #include <numeric/math/basis_lagrange.hpp>
  #include <numeric/math/fes/basis_h1.hpp>
  #include <numeric/math/fes/basis_l2.hpp>
  #include <numeric/equations/fem/diffusion_element_matrix.hpp>


  template <typename Scalar, typename ScalarMesh, typename ElementMatrix>
  __global__ void apply_matrix(
      ElementMatrix elem_mat,
      numeric::memory::ArrayConstView<numeric::dim_t, 1> group,
      numeric::memory::ArrayConstView<ScalarMesh, 2> vertices,
      numeric::memory::ArrayConstView<numeric::dim_t, 2> elements,
      numeric::memory::ArrayConstView<numeric::dim_t, 2> dofs,
      numeric::memory::ArrayConstView<Scalar, 1> u,
      numeric::memory::ArrayView<Scalar, 1> out) {
    static constexpr numeric::dim_t num_nodes = ElementMatrix::element_t::num_nodes;
    static constexpr numeric::dim_t num_basis_functions = ElementMatrix::num_basis_functions;
    const numeric::dim_t num_elements = group.shape(0);
    const numeric::dim_t world_dim = vertices.shape(0);

    const numeric::dim_t tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (tid >= num_elements) {
      return;
    }

    extern __shared__ Scalar work[];
    Scalar *local_work =
	work + hipThreadIdx_x * (world_dim * num_nodes + ElementMatrix::apply_work_size(world_dim) / sizeof(Scalar));
    Scalar (*nodes)[num_nodes] = reinterpret_cast<Scalar(*)[num_nodes]>(local_work);
    Scalar *apply_work = local_work + world_dim * num_nodes;

    Scalar elem_vec_in[num_basis_functions];  // Local u vector
    Scalar elem_vec_out[num_basis_functions]; // Output of local mat-vec

    const numeric::dim_t element = group(tid);
    
    // Extract physical coordinates of the current element's nodes
    for (numeric::dim_t node = 0 ; node < num_nodes ; ++node) {
      for (numeric::dim_t dim = 0 ; dim < world_dim ; ++dim) {
	nodes[dim][node] = vertices(dim, elements(node, element));
      }
    }

    // Gather the local dofs
    for (numeric::dim_t bf = 0; bf < num_basis_functions; ++bf) {
      const numeric::dim_t dof_idx = dofs(bf, element);
      elem_vec_in[bf] = u(dof_idx);
    }

    // Apply local element matrix
    elem_mat.apply(nodes, elem_vec_in, world_dim, elem_vec_out,
		   apply_work);

    // Scatter onto the global coefficient vector
    for (numeric::dim_t bf = 0; bf < num_basis_functions; ++bf) {
      const numeric::dim_t dof_idx = dofs(bf, element);
      out(dof_idx) += elem_vec_out[bf];
    }
  }
)";

hip::Kernel element_matrix_build_kernel(std::string_view scalar,
                                        std::string_view scalar_mesh,
                                        std::string_view element_matrix) {
  const std::string kernel_name = "apply_matrix<" + std::string(scalar) + ", " +
                                  std::string(scalar_mesh) + ", " +
                                  std::string(element_matrix) + ">";
  hip::Program program(kernel_src);
  program.instantiate_kernel(kernel_name);
  return program.get_kernel(kernel_name);
}

} // namespace internal

} // namespace numeric::equations::fem
