#include <iostream>
#include <numeric/io/netcdf_file.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_op.hpp>
#include <numeric/memory/linspace.hpp>
#include <numeric/memory/meshgrid.hpp>

using namespace numeric;

void minus_laplacian(memory::ArrayView<double, 2> &out,
                     const memory::ArrayConstView<double, 2> &in, double dx) {
  using sl = memory::Slice;
  const dim_t N = out.shape(0);
  out = 4 * in;
  out(sl(1, N), sl(0, N)) -= in(sl(0, N - 1), sl(0, N));
  out(sl(0, N - 1), sl(0, N)) -= in(sl(1, N), sl(0, N));
  out(sl(0, N), sl(1, N)) -= in(sl(0, N), sl(0, N - 1));
  out(sl(0, N), sl(0, N - 1)) -= in(sl(0, N), sl(1, N));
}

int main() {
  static constexpr numeric::dim_t N = 1024;

  const numeric::memory::Linspace<double> x(-1, 1, N);
  const auto [X, Y] = numeric::memory::meshgrid(x, x);
  const numeric::memory::Array<double, 2> f = X * X + Y * Y;

  auto file = io::NetCDFFile::create("fd_poisson.nc");
  const auto dim = file->create_dim("N", N);
  file->write("X", numeric::memory::Array<double, 2>(X), {dim, dim});
  file->write("Y", numeric::memory::Array<double, 2>(Y), {dim, dim});
  file->write("f", f, {dim, dim});

  return 0;
}
