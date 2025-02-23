#include <numeric/equations/fem/diffusion_element_matrix.hpp>
#include <numeric/equations/fem/finite_element_matrix.hpp>
#include <numeric/io/gmsh_reader.hpp>
#include <numeric/io/hdf5_file.hpp>
#include <numeric/io/vtkhdf_writer.hpp>
#include <numeric/math/conjugate_gradient.hpp>
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
  auto mesh =
      io::GmshReader<scalar_t, mesh::Tria<1>>::load(mesh_file, world_dim);
  std::cout << "Done reading " << mesh->num_vertices() << " vertices, "
            << mesh->num_elements<mesh::Tria<1>>() << " triangles" << std::endl;

  std::cout << "Constructing first-order H1 FE space" << std::endl;
  using mesh_t = mesh::UnstructuredMesh<scalar_t, mesh::Tria<1>>;
  using basis_t = math::fes::BasisH1<1>;
  using fes_t = math::fes::FESpace<basis_t, mesh_t>;
  fes_t fes(mesh);
  std::cout << "Done. Got " << fes.num_dofs() << " degrees of freedom."
            << std::endl;

  using element_matrix_factory_t =
      equations::fem::DiffusionElementMatrixFactory<scalar_t, basis_t>;
  using stiffness_matrix_t =
      equations::fem::FiniteElementMatrix<fes_t, element_matrix_factory_t>;
  stiffness_matrix_t lapl(fes);

  memory::Array<scalar_t, 1> f(memory::Shape<1>(fes.num_dofs()),
                               memory::MemoryType::HOST);
  memory::Array<scalar_t, 1> u(memory::Shape<1>(fes.num_dofs()),
                               memory::MemoryType::HOST);
  f = 0;
  u = 0;

  // TODO: This should be its own class somehow
  auto lapl_with_boundary = [&](const memory::ArrayConstView<scalar_t, 1> &x) {
    // TODO: Set boundary conditions here
  };
  // math::ConjugateGradient<scalar_t, stiffness_matrix_t> cg(lapl);
  // cg.set_tolerance(fes.num_dofs() * 1e-8);
  // cg.set_max_iterations(fes.num_dofs());
  // const auto [converged, num_iter, error] = cg.solve(f, u);

  io::VTKHDFWriter<fes_t> writer("fem_poisson.vtkhdf", fes);

  /*
  auto file = io::HDF5File::create("fem_poisson.vtkhdf");
  auto VTKHDF = file->create_group("VTKHDF");
  auto NumberOfConnectivityIds = VTKHDF->create_variable<dim_t>(
      "NumberOfConnectivityIds", memory::Shape<1>(1));
  auto NumberOfPoints =
      VTKHDF->create_variable<dim_t>("NumberOfPoints", memory::Shape<1>(1));
  auto NumberOfCells =
      VTKHDF->create_variable<dim_t>("NumberOfCells", memory::Shape<1>(1));
  auto Points = VTKHDF->create_variable<scalar_t>(
      "Points", memory::Shape<2>(fes.mesh()->num_vertices(), 3));
  auto Types = VTKHDF->create_variable<unsigned char>(
      "Types", memory::Shape<1>(fes.mesh()->num_elements<mesh::Tria<1>>()));
  auto Connectivity = VTKHDF->create_variable<dim_t>(
      "Connectivity",
      memory::Shape<1>(3 * fes.mesh()->num_elements<mesh::Tria<1>>()));
  auto Offsets = VTKHDF->create_variable<dim_t>(
      "Offsets",
      memory::Shape<1>(fes.mesh()->num_elements<mesh::Tria<1>>() + 1));
  VTKHDF->write_attribute("Type", "UnstructuredGrid");
  VTKHDF->write_attribute("Version", std::vector<int>{2, 2});
  const dim_t num_conn_ids = 3 * fes.mesh()->num_elements<mesh::Tria<1>>();
  NumberOfConnectivityIds->write(memory::ArrayConstView<dim_t, 1>(
      &num_conn_ids, memory::Shape<1>(1), memory::MemoryType::HOST));
  const dim_t num_points = fes.mesh()->num_vertices();
  NumberOfPoints->write(memory::ArrayConstView<dim_t, 1>(
      &num_points, memory::Shape<1>(1), memory::MemoryType::HOST));
  const dim_t num_cells = fes.mesh()->num_elements<mesh::Tria<1>>();
  NumberOfCells->write(memory::ArrayConstView<dim_t, 1>(
      &num_cells, memory::Shape<1>(1), memory::MemoryType::HOST));
  const auto points = fes.mesh()->vertices();
  memory::Array<scalar_t, 2> points_flat(
      memory::Shape<2>(fes.mesh()->num_vertices(), 3),
      memory::MemoryType::HOST);
  points_flat(memory::Slice(), 0) = points(0);
  if (fes.mesh()->world_dim() > 1) {
    points_flat(memory::Slice(), 1) = points(1);
  } else {
    points_flat(memory::Slice(), 1) = 0;
  }
  if (fes.mesh()->world_dim() > 2) {
    points_flat(memory::Slice(), 2) = points(2);
  } else {
    points_flat(memory::Slice(), 2) = 0;
  }
  Points->write(points_flat);
  memory::Array<unsigned char, 1> types(
      memory::Shape<1>(fes.mesh()->num_elements<mesh::Tria<1>>()),
      memory::MemoryType::HOST);
  types = 5; // All are triangles
  Types->write(types);
  memory::Array<dim_t, 1> connectivity_flat(
      memory::Shape<1>(3 * fes.mesh()->num_elements<mesh::Tria<1>>()),
      memory::MemoryType::HOST);
  const auto elements = fes.mesh()->get_elements<mesh::Tria<1>>();
  connectivity_flat(memory::Slice(0, -1, 3)) = elements(0);
  connectivity_flat(memory::Slice(1, -1, 3)) = elements(1);
  connectivity_flat(memory::Slice(2, -1, 3)) = elements(2);
  Connectivity->write(connectivity_flat);
  memory::Array<dim_t, 1> offsets(
      memory::Shape<1>(fes.mesh()->num_elements<mesh::Tria<1>>() + 1),
      memory::MemoryType::HOST);
  for (dim_t i = 0; i < offsets.shape(0); ++i) {
    offsets(i) = 3 * i;
  }
  Offsets->write(offsets);
  */

  return 0;
}
