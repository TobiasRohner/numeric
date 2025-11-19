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
#include <numeric/math/linear_system.hpp>
#include <numeric/math/mesh_function.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/mesh/elements.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/utils/lambda.hpp>

using namespace numeric;

int main(int argc, char *argv[]) {
  using scalar_t = double;
  static constexpr dim_t world_dim = 2;
  const memory::MemoryType memory_type = memory::MemoryType::DEVICE;

  using mesh_t = mesh::UnstructuredMesh<scalar_t, mesh::Tria<1>>;
  using basis_t = math::fes::BasisH1<4>;
  using fes_t = math::fes::FESpace<basis_t, mesh_t>;
  using element_matrix_factory_t =
      equations::fem::DiffusionElementMatrixFactory<scalar_t, basis_t>;
  using stiffness_matrix_t =
      equations::fem::FiniteElementMatrix<fes_t, element_matrix_factory_t>;
  using element_vector_factory_t =
      equations::fem::LoadElementVectorFactory<scalar_t, basis_t>;
  using load_vector_t =
      equations::fem::FiniteElementVector<fes_t, element_vector_factory_t>;

  const std::string mesh_file = argv[1];

  std::cout << "Reading mesh " << mesh_file << std::endl;
  std::shared_ptr<mesh_t> mesh =
      io::GmshReader<scalar_t, mesh::Tria<1>>::load(mesh_file, world_dim);
  std::cout << "Done reading " << mesh->num_vertices() << " vertices, "
            << mesh->num_elements<mesh::Tria<1>>() << " triangles" << std::endl;

  std::cout << "Constructing H1 FE space of order " << fes_t::basis_t::order
            << std::endl;
  auto fes = std::make_shared<fes_t>(mesh);
  std::cout << "Done. Got " << fes->num_dofs() << " degrees of freedom."
            << std::endl;

  // TODO: Figure out real boundary dofs
  memory::Array<dim_t, 1> dirichlet_dofs(memory::Shape<1>(1),
                                         memory::MemoryType::HOST);
  dirichlet_dofs(0) = 0;

  auto lapl = std::make_shared<stiffness_matrix_t>(fes);
  load_vector_t load(fes);

  math::LinearSystem<scalar_t> system(lapl);
  system.to(memory_type);
  system.set_fixed_dofs(dirichlet_dofs);
  auto cg = std::make_shared<math::ConjugateGradient<scalar_t>>();
  cg->set_tolerance(1e-8);
  cg->set_max_iterations(fes->num_dofs());
  system.set_solver(cg);

  const auto f = NUMERIC_LAMBDA(
      [](auto *x) -> ::numeric::meta::remove_cvref_t<decltype(*x)> {
        const scalar_t x1 = x[0];
        const scalar_t x2 = x[1];
        const scalar_t r2 = x1 * x1 + x2 * x2;
        return (r2 < 0.01 ? 1 : 0) - 0.01;
      });
  load.assemble(f, system.rhs());

  math::MeshFunction<scalar_t, fes_t> u(fes);
  math::MeshFunction<scalar_t, fes_t> offset(fes);
  offset.dofs() = 100;
  system.solve(u.dofs(), offset.dofs());
  const auto [converged, num_iter, error] = cg->result();
  if (converged) {
    std::cout << "Converged in " << num_iter << " iterations (error = " << error
              << ")" << std::endl;
  } else {
    std::cout << "No convergence in " << num_iter
              << " iterations (error = " << error << ")" << std::endl;
  }

  io::VTKHDFWriter<basis_t::order, io::VTKHDFFunctionSpaceType::CONTINUOUS,
                   mesh_t>
      writer("fem_poisson.vtkhdf", mesh);
  writer.write("u", u);
  math::MeshFunction<scalar_t, fes_t> Au(fes);
  Au.dofs() = (*lapl)(u.dofs());
  writer.write("Au", Au);
  math::MeshFunction<scalar_t, fes_t> mfrhs(fes);
  mfrhs.dofs() = system.rhs();
  writer.write("f", mfrhs);

  return 0;
}
