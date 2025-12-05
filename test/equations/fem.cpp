#include <filesystem>
#include <gtest/gtest.h>
#include <numeric/equations/fem/diffusion_element_matrix.hpp>
#include <numeric/equations/fem/finite_element_matrix.hpp>
#include <numeric/equations/fem/finite_element_vector.hpp>
#include <numeric/equations/fem/load_element_vector.hpp>
#include <numeric/io/gmsh_reader.hpp>
#include <numeric/math/fes/basis_h1.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>
#include <numeric/utils/lambda.hpp>

static auto load_unit_disk_tria() {
  std::filesystem::path path = numeric::DATA_DIR;
  path /= "test/unit_disk_tria.msh";
  return numeric::io::GmshReader<double, numeric::mesh::Tria<1>>::load(
      path.c_str(), 2);
}

TEST(fem, stiffness_matrix_null_space) {
  using mesh_t =
      numeric::mesh::UnstructuredMesh<double, numeric::mesh::Tria<1>>;
  using basis_t = numeric::math::fes::BasisH1<4>;
  using fes_t = numeric::math::fes::FESpace<basis_t, mesh_t>;
  using element_matrix_factory_t =
      numeric::equations::fem::DiffusionElementMatrixFactory<double, basis_t>;
  using stiffness_matrix_t =
      numeric::equations::fem::FiniteElementMatrix<fes_t,
                                                   element_matrix_factory_t>;

  std::shared_ptr<mesh_t> mesh = load_unit_disk_tria();
  std::shared_ptr<fes_t> fes = std::make_shared<fes_t>(mesh);
  stiffness_matrix_t A(fes);

  numeric::memory::Array<double, 1> x(
      numeric::memory::Shape<1>(fes->num_dofs()),
      numeric::memory::MemoryType::HOST);
  numeric::memory::Array<double, 1> out(
      numeric::memory::Shape<1>(fes->num_dofs()),
      numeric::memory::MemoryType::HOST);
  x = 1;
  A(x, out);

  for (numeric::dim_t i = 0; i < fes->num_dofs(); ++i) {
    ASSERT_NEAR(out(i), 0, 1e-8);
  }
}

#if NUMERIC_ENABLE_HIP
TEST(fem, load_vector_device) {
  using mesh_t =
      numeric::mesh::UnstructuredMesh<double, numeric::mesh::Tria<1>>;
  using basis_t = numeric::math::fes::BasisH1<4>;
  using fes_t = numeric::math::fes::FESpace<basis_t, mesh_t>;
  using element_vector_factory_t =
      numeric::equations::fem::LoadElementVectorFactory<double, basis_t>;
  using load_vector_t =
      numeric::equations::fem::FiniteElementVector<fes_t,
                                                   element_vector_factory_t>;

  std::shared_ptr<mesh_t> mesh = load_unit_disk_tria();
  std::shared_ptr<fes_t> fes = std::make_shared<fes_t>(mesh);
  load_vector_t load(fes);

  const auto f = NUMERIC_LAMBDA([](double *x) -> double {
    const double x1 = x[0];
    const double x2 = x[1];
    const double r2 = x1 * x1 + x2 * x2;
    return (r2 < 0.01 ? 1 : 0) - 0.01;
  });

  numeric::memory::Array<double, 1> rhs_host(
      numeric::memory::Shape<1>(fes->num_dofs()),
      numeric::memory::MemoryType::HOST);
  numeric::memory::Array<double, 1> rhs_device(
      numeric::memory::Shape<1>(fes->num_dofs()),
      numeric::memory::MemoryType::DEVICE);
  load.assemble(f, rhs_host);
  load.to(numeric::memory::MemoryType::DEVICE);
  load.assemble(f, rhs_device);

  rhs_device.to(numeric::memory::MemoryType::HOST);
  for (numeric::dim_t i = 0; i < fes->num_dofs(); ++i) {
    ASSERT_NEAR(rhs_host(i), rhs_device(i), 1e-8);
  }
}

TEST(fem, stiffness_matrix_device) {
  using mesh_t =
      numeric::mesh::UnstructuredMesh<double, numeric::mesh::Tria<1>>;
  using basis_t = numeric::math::fes::BasisH1<4>;
  using fes_t = numeric::math::fes::FESpace<basis_t, mesh_t>;
  using element_matrix_factory_t =
      numeric::equations::fem::DiffusionElementMatrixFactory<double, basis_t>;
  using stiffness_matrix_t =
      numeric::equations::fem::FiniteElementMatrix<fes_t,
                                                   element_matrix_factory_t>;

  std::shared_ptr<mesh_t> mesh = load_unit_disk_tria();
  std::shared_ptr<fes_t> fes = std::make_shared<fes_t>(mesh);
  stiffness_matrix_t A(fes);

  numeric::memory::Array<double, 2> A_host(
      numeric::memory::Shape<2>(fes->num_dofs(), fes->num_dofs()),
      numeric::memory::MemoryType::HOST);
  numeric::memory::Array<double, 2> A_device(
      numeric::memory::Shape<2>(fes->num_dofs(), fes->num_dofs()),
      numeric::memory::MemoryType::DEVICE);

  numeric::memory::Array<double, 1> x_host(
      numeric::memory::Shape<1>(fes->num_dofs()),
      numeric::memory::MemoryType::HOST);
  numeric::memory::Array<double, 1> x_device(
      numeric::memory::Shape<1>(fes->num_dofs()),
      numeric::memory::MemoryType::DEVICE);
  for (numeric::dim_t i = 0; i < fes->num_dofs(); ++i) {
    x_host = 0;
    x_host(i) = 1;
    A(x_host, A_host(i));
  }
  A.to(numeric::memory::MemoryType::DEVICE);
  for (numeric::dim_t i = 0; i < fes->num_dofs(); ++i) {
    x_host = 0;
    x_host(i) = 1;
    x_device = x_host;
    A(x_device, A_device(i));
  }

  A_device.to(numeric::memory::MemoryType::HOST);
  for (numeric::dim_t i = 0; i < fes->num_dofs(); ++i) {
    for (numeric::dim_t j = 0; j < fes->num_dofs(); ++j) {
      ASSERT_NEAR(A_host(i, j), A_device(i, j), 1e-8);
    }
  }
}
#endif
