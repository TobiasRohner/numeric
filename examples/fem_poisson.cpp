#include <numeric/equations/fem/diffusion_element_matrix.hpp>
#include <numeric/equations/fem/finite_element_matrix.hpp>
#include <numeric/io/gmsh_reader.hpp>
#include <numeric/io/hdf5_file.hpp>
#include <numeric/io/vtkhdf_writer.hpp>
#include <numeric/math/conjugate_gradient.hpp>
#include <numeric/math/fes/basis_h1.hpp>
#include <numeric/math/fes/fe_space.hpp>
#include <numeric/math/mesh_function.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/mesh/elements.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>

using namespace numeric;

int main(int argc, char *argv[]) {
  using scalar_t = double;
  static constexpr dim_t world_dim = 2;

  const std::string mesh_file = argv[1];

  std::cout << "Reading mesh " << mesh_file << std::endl;
  auto mesh =
      io::GmshReader<scalar_t, mesh::Tria<1>>::load(mesh_file, world_dim);
  std::cout << "Done reading " << mesh->num_vertices() << " vertices, "
            << mesh->num_elements<mesh::Tria<1>>() << " triangles" << std::endl;

  std::cout << "Constructing first-order H1 FE space" << std::endl;
  using mesh_t = mesh::UnstructuredMesh<scalar_t, mesh::Tria<1>>;
  using basis_t = math::fes::BasisH1<1>;
  using fes_t = math::fes::FESpace<basis_t, mesh_t>;
  auto fes = std::make_shared<fes_t>(mesh);
  std::cout << "Done. Got " << fes->num_dofs() << " degrees of freedom."
            << std::endl;

  using element_matrix_factory_t =
      equations::fem::DiffusionElementMatrixFactory<scalar_t, basis_t>;
  using stiffness_matrix_t =
      equations::fem::FiniteElementMatrix<fes_t, element_matrix_factory_t>;
  stiffness_matrix_t lapl(*fes);

  math::MeshFunction<scalar_t, fes_t> f(fes);
  math::MeshFunction<scalar_t, fes_t> u(fes);

  // TODO: This should be its own class somehow
  auto lapl_with_boundary = [&](const memory::ArrayConstView<scalar_t, 1> &x) {
    // TODO: Set boundary conditions here
  };
  // math::ConjugateGradient<scalar_t, stiffness_matrix_t> cg(lapl);
  // cg.set_tolerance(fes.num_dofs() * 1e-8);
  // cg.set_max_iterations(fes.num_dofs());
  // const auto [converged, num_iter, error] = cg.solve(f, u);

  io::VTKHDFWriter<basis_t::order, io::VTKHDFFunctionSpaceType::CONTINUOUS,
                   mesh_t>
      writer("fem_poisson.vtkhdf", mesh);
  writer.write("f", f);
  writer.write("u", u);

  return 0;
}
