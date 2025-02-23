#include <numeric/io/gmsh_reader.hpp>
#include <numeric/math/fes/basis_h1.hpp>
#include <numeric/math/fes/basis_l2.hpp>
#include <numeric/math/fes/fe_space.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/mesh/elements.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>
#include <numeric/utils/tuple.hpp>

using namespace numeric;

template <typename Element>
void print_elements(const memory::ArrayConstView<dim_t, 2> &elements) {
  std::cout << Element::name << '\n';
  for (dim_t node = 0; node < Element::num_nodes; ++node) {
    for (dim_t element = 0; element < elements.shape(1); ++element) {
      std::cout << '\t' << elements(node, element);
    }
    std::cout << '\n';
  }
}

template <typename Scalar, typename... ElementTypes>
void print_mesh(const mesh::UnstructuredMesh<Scalar, ElementTypes...> &mesh) {
  std::cout << mesh.world_dim() << "D mesh with " << mesh.num_vertices()
            << " vertices\n";
  std::cout << "Vertices:\n";
  for (dim_t dim = 0; dim < mesh.world_dim(); ++dim) {
    for (dim_t vertex = 0; vertex < mesh.num_vertices(); ++vertex) {
      std::cout << '\t' << mesh.vertices()(dim, vertex);
    }
    std::cout << '\n';
  }
  (print_elements<ElementTypes>(mesh.template get_elements<ElementTypes>()),
   ...);
}

template <typename Element>
void print_dofs(const memory::ArrayConstView<dim_t, 2> &dofs) {
  std::cout << Element::name << '\n';
  for (dim_t dof = 0; dof < dofs.shape(0); ++dof) {
    for (dim_t element = 0; element < dofs.shape(1); ++element) {
      std::cout << '\t' << dofs(dof, element);
    }
    std::cout << '\n';
  }
}

int main(int argc, char *argv[]) {
  const dim_t world_dim = 2;
  auto mesh = io::GmshReader<double, mesh::Tria<1>, mesh::Quad<1>>::load(
      argv[1], world_dim);
  print_mesh(*mesh);

  math::fes::FESpace<math::fes::BasisH1<2>,
                     meta::remove_cvref_t<decltype(*mesh)>>
      fes(mesh);

  std::cout << "Finite Element Space has " << fes.num_dofs()
            << " degrees of freedom.\n";
  print_dofs<mesh::Tria<1>>(fes.template dof_map<mesh::Tria<1>>());
  print_dofs<mesh::Quad<1>>(fes.template dof_map<mesh::Quad<1>>());

  return 0;
}
