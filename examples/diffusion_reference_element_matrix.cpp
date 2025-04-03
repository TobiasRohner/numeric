#include <iomanip>
#include <numeric/equations/fem/diffusion_element_matrix.hpp>
#include <numeric/equations/fem/finite_element_matrix.hpp>
#include <numeric/math/fes/basis_h1.hpp>
#include <numeric/math/fes/fe_space.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/mesh/elements.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>

using namespace numeric;

using scalar_t = double;

template <typename Element>
std::shared_ptr<mesh::UnstructuredMesh<scalar_t, Element>> generate_mesh() {
  using ref_el_t = typename Element::ref_el_t;
  static constexpr dim_t world_dim = Element::dim;
  static constexpr dim_t num_vertices = Element::num_nodes;
  auto mesh = std::make_shared<mesh::UnstructuredMesh<scalar_t, Element>>(
      world_dim, num_vertices, 1);
  auto vertices = mesh->vertices();
  auto elements = mesh->template get_elements<Element>();
  scalar_t ref_nodes[num_vertices][world_dim];
  Element::get_ref_nodes(ref_nodes);
  for (dim_t node = 0; node < num_vertices; ++node) {
    for (dim_t i = 0; i < world_dim; ++i) {
      vertices(i, node) = ref_nodes[node][i];
    }
    elements(node, 0) = node;
  }
  return mesh;
}

template <typename Element, dim_t Order>
memory::Array<scalar_t, 2> diffusion_reference_element_matrix() {
  using mesh_t = mesh::UnstructuredMesh<scalar_t, Element>;
  using basis_t = math::fes::BasisH1<Order>;
  using fes_t = math::fes::FESpace<basis_t, mesh_t>;
  const auto mesh = generate_mesh<Element>();
  fes_t fes(mesh);
  using element_matrix_factory_t =
      equations::fem::DiffusionElementMatrixFactory<scalar_t, basis_t>;
  equations::fem::FiniteElementMatrix<fes_t, element_matrix_factory_t> lapl(
      fes);
  memory::Array<scalar_t, 1> u(memory::Shape<1>(fes.num_dofs()),
                               memory::MemoryType::HOST);
  memory::Array<scalar_t, 2> A(memory::Shape<2>(fes.num_dofs(), fes.num_dofs()),
                               memory::MemoryType::HOST);
  for (dim_t i = 0; i < fes.num_dofs(); ++i) {
    u = 0;
    u(i) = 1;
    lapl.apply(fes, u, A(i));
  }
  return A;
}

template <typename Element, dim_t Order> void print_matrix() {
  using ref_el_t = typename Element::ref_el_t;
  const auto A = diffusion_reference_element_matrix<Element, Order>();
  const dim_t N = A.shape(0);
  std::cout << "Element Matrix of order " << Order << " on the "
            << ref_el_t::name << std::endl;
  for (dim_t i = 0; i < N; ++i) {
    for (dim_t j = 0; j < N; ++j) {
      std::cout << A(i, j) << '\t';
    }
    std::cout << '\n';
  }
}

int main(int argc, char *argv[]) {
  print_matrix<mesh::Segment<1>, 1>();
  print_matrix<mesh::Segment<1>, 2>();
  print_matrix<mesh::Segment<1>, 3>();
  print_matrix<mesh::Segment<1>, 4>();
  print_matrix<mesh::Segment<1>, 5>();
  print_matrix<mesh::Tria<1>, 1>();
  print_matrix<mesh::Tria<1>, 2>();
  print_matrix<mesh::Tria<1>, 3>();
  print_matrix<mesh::Quad<1>, 1>();
  print_matrix<mesh::Quad<1>, 2>();
  print_matrix<mesh::Quad<1>, 3>();
  print_matrix<mesh::Tetra<1>, 1>();
  print_matrix<mesh::Tetra<1>, 2>();
  print_matrix<mesh::Cube<1>, 1>();
  print_matrix<mesh::Cube<1>, 2>();
  return 0;
}
