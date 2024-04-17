#include <iostream>
#include <numeric/io/netcdf_file.hpp>
#include <numeric/math/conjugate_gradient.hpp>
#include <numeric/math/sum.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_op.hpp>
#include <numeric/memory/constant.hpp>
#include <numeric/memory/linspace.hpp>
#include <numeric/memory/meshgrid.hpp>
#include <numeric/mesh/regular_grid.hpp>

using namespace numeric;

using real_t = float;

int main() {
  using sl = memory::Slice;

  const memory::MemoryType memory_type = memory::MemoryType::HOST;

  static constexpr dim_t N = 1024;
  const real_t tol = 1e-5;

  mesh::RegularGrid<real_t, 2> mesh(memory::Shape<2>(N, N), 1, memory_type);
  mesh.set_origin(-1, -1);
  mesh.set_size(2, 2);
  const real_t dx = mesh.dx(0);
  const real_t dy = mesh.dx(1);

  const auto [X, Y] = mesh.cell_positions();
  const memory::Array<real_t, 2> u_exact = X * X + Y * Y;
  const memory::Array<real_t, 2> f = memory::Constant<real_t, 2>(
      memory::Shape<2>(N + 2, N + 2), -4, memory_type);

  memory::Array<real_t, 2> u(memory::Shape<2>(N + 2, N + 2), memory_type);
  u = u_exact;
  u(sl(1, -2), sl(1, -2)) = 0;

  memory::Array<real_t, 2> bd = u_exact;
  auto negative_laplacian = [&](memory::ArrayConstView<real_t, 2> &x) {
    bd(sl(1, -2), sl(1, -2)) = x;
    return (4 * bd(sl(1, -2), sl(1, -2)) - bd(sl(1, -2), sl(0, -3)) -
            bd(sl(1, -2), sl(2, -1)) - bd(sl(0, -3), sl(1, -2)) -
            bd(sl(2, -1), sl(1, -2))) /
           (dx * dy);
  };
  math::ConjugateGradient<real_t, decltype(negative_laplacian)> cg(
      negative_laplacian);
  cg.set_tolerance(N * N * 1e-8);
  cg.set_max_iterations(N * N);
  const auto [converged, num_iter, error] =
      cg.solve(f(sl(1, -2), sl(1, -2)), u(sl(1, -2), sl(1, -2)));
  if (converged) {
    std::cout << "Converged in " << num_iter << " iterations (error = " << error
              << ")" << std::endl;
  } else {
    std::cout << "No convergence in " << num_iter
              << " iterations (error = " << error << ")" << std::endl;
  }
  std::cout << "L2-err = " << math::norm::l2(u - u_exact) / (N * N)
            << std::endl;

  memory::Array<real_t, 2> f_host(f.shape());
  memory::Array<real_t, 2> u_host(u.shape());
  memory::Array<real_t, 2> u_exact_host(u_exact.shape());
  memory::copy(f_host, f);
  memory::copy(u_host, u);
  memory::copy(u_exact_host, u_exact);
  auto file = io::NetCDFFile::create("fd_poisson.nc");
  const auto dim_N = file->create_dim("N", N + 2);
  file->write("f", f_host, {dim_N, dim_N});
  file->write("u", u_host, {dim_N, dim_N});
  file->write("u_exact", u_exact_host, {dim_N, dim_N});

  return 0;
}
