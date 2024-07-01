#include <iomanip>
#include <numeric/equations/fem/diffusion_element_matrix.hpp>
#include <numeric/equations/fem/finite_element_matrix.hpp>
#include <numeric/io/gmsh_reader.hpp>
#include <numeric/math/fes/basis_h1.hpp>
#include <numeric/math/fes/fe_space.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/mesh/elements.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>

using namespace numeric;

int main(int argc, char *argv[]) {
  using scalar_t = double;
  static constexpr dim_t world_dim = 2;

  const std::string mesh_file = argv[1];

  std::cout << "Reading mesh " << mesh_file << std::endl;
  auto mesh = io::GmshReader<scalar_t, mesh::Tria<1>, mesh::Quad<1>>::load(
      mesh_file, world_dim);
  std::cout << "Done reading " << mesh.num_vertices() << " vertices, "
            << mesh.num_elements<mesh::Tria<1>>() << " triangles, and "
            << mesh.num_elements<mesh::Quad<1>>() << " quads." << std::endl;

  std::cout << "Constructing first-order H1 FE space" << std::endl;
  using mesh_t = mesh::UnstructuredMesh<scalar_t, mesh::Tria<1>, mesh::Quad<1>>;
  using basis_t = math::fes::BasisH1<1>;
  using fes_t = math::fes::FESpace<basis_t, mesh_t>;
  fes_t fes(std::move(mesh));
  std::cout << "Done. Got " << fes.num_dofs() << " degrees of freedom."
            << std::endl;

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

  for (dim_t i = 0; i < fes.num_dofs(); ++i) {
    for (dim_t j = 0; j < fes.num_dofs(); ++j) {
      std::cout << std::showpos << std::left << std::setw(9) << A(i, j) << " ";
    }
    std::cout << "\n";
  }

  return 0;
}
