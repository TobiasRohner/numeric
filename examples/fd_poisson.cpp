#include <iostream>
#include <numeric/io/netcdf_file.hpp>
#include <numeric/math/sum.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_op.hpp>
#include <numeric/memory/constant.hpp>
#include <numeric/memory/linspace.hpp>
#include <numeric/memory/meshgrid.hpp>

using namespace numeric;

int main() {
  using sl = memory::Slice;

  static constexpr dim_t N = 128;
  const double dx = 2. / N;

  const memory::Linspace<double> x(-1 - dx, 1 + dx, N + 2);
  const auto [X, Y] = memory::meshgrid(x, x);
  const memory::Array<double, 2> u_exact = X * X + Y * Y;
  const memory::Constant<double, 2> f(memory::Shape<2>(N + 2, N + 2), -4);

  memory::Array<double, 2> u(memory::Shape<2>(N + 2, N + 2));
  memory::Array<double, 2> tmp(memory::Shape<2>(N + 2, N + 2));
  u = u_exact;
  u(sl(1, -2), sl(1, -2)) = 0;
  tmp = u_exact;

  auto *xi = &u;
  auto *xip1 = &tmp;
  double norm = 0;
  do {
    (*xip1)(sl(1, -2), sl(1, -2)) =
        0.25 * (dx * dx * f(sl(1, -2), sl(1, -2)) +
                (*xi)(sl(1, -2), sl(0, -3)) + (*xi)(sl(1, -2), sl(2, -1)) +
                (*xi)(sl(0, -3), sl(1, -2)) + (*xi)(sl(2, -1), sl(1, -2)));
    std::swap(xi, xip1);
    double norm2 = math::sum(memory::pow<2>(*xip1 - *xi));
    norm = 4 * std::sqrt(norm2) / ((N + 2) * (N + 2));
  } while (norm > 1e-8);
  std::cout << "Converged with relative error of " << norm << std::endl;

  auto file = io::NetCDFFile::create("fd_poisson.nc");
  const auto dim = file->create_dim("N", N + 2);
  file->write("f", memory::Array<double, 2>(f), {dim, dim});
  file->write("u", *xi, {dim, dim});
  file->write("u_exact", u_exact, {dim, dim});

  return 0;
}
