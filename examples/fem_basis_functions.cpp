#include <numeric/io/gmsh_reader.hpp>
#include <numeric/io/hdf5_file.hpp>
#include <numeric/io/vtkhdf_writer.hpp>
#include <numeric/math/fes/basis_h1.hpp>
#include <numeric/math/fes/fe_space.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/mesh/elements.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>

using namespace numeric;

int main(int argc, char *argv[]) {
  using scalar_t = double;
  static constexpr dim_t world_dim = 2;
  using mesh_t = mesh::UnstructuredMesh<scalar_t, mesh::Tria<1>>;
  using basis_t = math::fes::BasisH1<3>;
  using fes_t = math::fes::FESpace<basis_t, mesh_t>;

  const std::string mesh_file = argv[1];

  std::cout << "Reading mesh " << mesh_file << std::endl;
  std::shared_ptr<mesh_t> mesh =
      io::GmshReader<scalar_t, mesh::Tria<1>>::load(mesh_file, world_dim);
  std::cout << "Done reading " << mesh->num_vertices() << " vertices, "
            << mesh->num_elements<mesh::Tria<1>>() << " triangles" << std::endl;

  std::cout << "Constructing H1 FE space" << std::endl;
  auto fes = std::make_shared<fes_t>(mesh);
  std::cout << "Done. Got " << fes->num_dofs() << " degrees of freedom."
            << std::endl;

  io::VTKHDFWriter<basis_t::order, io::VTKHDFFunctionSpaceType::CONTINUOUS,
                   mesh_t>
      writer("fem_basis_functions.vtkhdf", mesh);

  for (dim_t i = 0; i < fes->num_dofs(); ++i) {
    math::MeshFunction<scalar_t, fes_t> u(fes);
    u.dofs() = 0;
    u.dofs()(i) = 1;
    writer.write("basis_" + std::to_string(i), u);
  }

  return 0;
}
