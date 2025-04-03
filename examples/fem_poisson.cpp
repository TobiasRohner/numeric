#include <numeric/equations/fem/diffusion_element_matrix.hpp>
#include <numeric/equations/fem/finite_element_matrix.hpp>
#include <numeric/equations/fem/finite_element_vector.hpp>
#include <numeric/equations/fem/load_element_vector.hpp>
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
  using basis_t = math::fes::BasisH1<3>;
  using fes_t = math::fes::FESpace<basis_t, mesh_t>;
  auto fes = std::make_shared<fes_t>(mesh);
  std::cout << "Done. Got " << fes->num_dofs() << " degrees of freedom."
            << std::endl;

  using element_matrix_factory_t =
      equations::fem::DiffusionElementMatrixFactory<scalar_t, basis_t>;
  using stiffness_matrix_t =
      equations::fem::FiniteElementMatrix<fes_t, element_matrix_factory_t>;
  using element_vector_factory_t =
      equations::fem::LoadElementVectorFactory<scalar_t, basis_t>;
  using load_vector_t =
      equations::fem::FiniteElementVector<fes_t, element_vector_factory_t>;
  stiffness_matrix_t lapl(*fes);
  load_vector_t load(*fes);

  const auto f = [](scalar_t *x) -> scalar_t {
    const scalar_t x1 = x[0];
    const scalar_t x2 = x[1];
    const scalar_t r2 = x1 * x1 + x2 * x2;
    return r2 < 0.01 ? 1 : 0;
  };
  memory::Array<scalar_t, 1> rhs(memory::Shape<1>(fes->num_dofs() + 1),
                                 memory::MemoryType::HOST);
  load.assemble(*fes, f, rhs(memory::Slice(0, -2)));
  rhs(rhs.size() - 1) = 0;

  math::MeshFunction<scalar_t, fes_t> u(fes);
  u.dofs() = 0;

  // TODO: This should be its own class somehow
  memory::Array<scalar_t, 1> x(memory::Shape<1>(fes->num_dofs() + 1),
                               memory::MemoryType::HOST);
  memory::Array<scalar_t, 1> Ax(memory::Shape<1>(fes->num_dofs() + 1),
                                memory::MemoryType::HOST);
  auto lapl_with_boundary = [&](const memory::ArrayConstView<scalar_t, 1> &x) {
    lapl.apply(*fes, x(memory::Slice(0, -2)), Ax(memory::Slice(0, -2)));
    Ax(memory::Slice(0, -2)) -= x(x.size() - 1) / (x.size() - 1);
    Ax(x.size() - 1) =
        x(x.size() - 1) - math::sum(x(memory::Slice(0, -2))) / (x.size() - 1);
    return Ax.const_view();
  };
  math::ConjugateGradient<scalar_t, decltype(lapl_with_boundary)> cg(
      lapl_with_boundary);
  cg.set_tolerance(fes->num_dofs() * 1e-8);
  cg.set_max_iterations(fes->num_dofs());
  x = 0;
  const auto [converged, num_iter, error] = cg.solve(rhs, x);
  if (converged) {
    std::cout << "Converged in " << num_iter << " iterations (error = " << error
              << ")" << std::endl;
  } else {
    std::cout << "No convergence in " << num_iter
              << " iterations (error = " << error << ")" << std::endl;
  }
  u.dofs() = x(memory::Slice(0, -2));

  io::VTKHDFWriter<basis_t::order, io::VTKHDFFunctionSpaceType::CONTINUOUS,
                   mesh_t>
      writer("fem_poisson.vtkhdf", mesh);
  writer.write("u", u);

  return 0;
}
