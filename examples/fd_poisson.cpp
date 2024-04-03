#include <iostream>
#include <numeric/io/netcdf_file.hpp>
#include <numeric/math/sum.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_op.hpp>
#include <numeric/memory/constant.hpp>
#include <numeric/memory/linspace.hpp>
#include <numeric/memory/meshgrid.hpp>

using namespace numeric;

static decltype(auto)
negative_laplacian(double dx, const memory::ArrayConstView<double, 2> &u) {
  using sl = memory::Slice;
  return (4 * u(sl(1, -2), sl(1, -2)) - u(sl(1, -2), sl(0, -3)) -
          u(sl(1, -2), sl(2, -1)) - u(sl(0, -3), sl(1, -2)) -
          u(sl(2, -1), sl(1, -2))) /
         (dx * dx);
}

int main() {
  using sl = memory::Slice;

  const memory::MemoryType memory_type = memory::MemoryType::DEVICE;

  static constexpr dim_t N = 1024;
  const double dx = 2. / (N - 1);
  const double tol = 1e-5;

  const memory::Linspace<double> x(-1 - dx, 1 + dx, N + 2, true, memory_type);
  const auto [X, Y] = memory::meshgrid(x, x);
  const memory::Array<double, 2> u_exact = X * X + Y * Y;
  const memory::Constant<double, 2> f(memory::Shape<2>(N + 2, N + 2), -4,
                                      memory_type);

  memory::Array<double, 2> u(memory::Shape<2>(N + 2, N + 2), memory_type);
  u = u_exact;
  u(sl(1, -2), sl(1, -2)) = 0;

  memory::Array<double, 2> r =
      f(sl(1, -2), sl(1, -2)) - negative_laplacian(dx, u);
  memory::Array<double, 2> p(u.shape(), memory_type);
  p = u;
  p(sl(1, -2), sl(1, -2)) = r;
  memory::Array<double, 2> Ap(memory::Shape<2>(N, N), memory_type);
  std::vector<double> norms;
  for (dim_t i = 0; i < N * N; ++i) {
    Ap = negative_laplacian(dx, p);
    const double rk2 = math::sum(memory::pow<2>(r));
    const double alpha = rk2 / math::sum(p(sl(1, -2), sl(1, -2)) * Ap);
    u(sl(1, -2), sl(1, -2)) += alpha * p(sl(1, -2), sl(1, -2));
    r -= alpha * Ap;
    // r = f(sl(1,-2),sl(1,-2)) - negative_laplacian(dx, u);
    const double rkp12 = math::sum(memory::pow<2>(r));
    const double norm = std::sqrt(rkp12) / (N * N);
    norms.push_back(norm);
    std::cout << norm << std::endl;
    if (norm < tol) {
      std::cout << "Converged in " << i + 1 << " iterations with norm " << norm
                << std::endl;
      break;
    }
    const double beta = rkp12 / rk2;
    p(sl(1, -2), sl(1, -2)) = r + beta * p(sl(1, -2), sl(1, -2));
  }

  memory::Array<double, 2> f_host(f.shape());
  memory::Array<double, 2> u_host(u.shape());
  memory::Array<double, 2> u_exact_host(u_exact.shape());
  memory::copy(f_host, f);
  memory::copy(u_host, u);
  memory::copy(u_exact_host, u_exact);
  auto file = io::NetCDFFile::create("fd_poisson.nc");
  const auto dim_N = file->create_dim("N", N + 2);
  const auto dim_M = file->create_dim("M", norms.size());
  file->write("f", f_host, {dim_N, dim_N});
  file->write("u", u_host, {dim_N, dim_N});
  file->write("u_exact", u_exact_host, {dim_N, dim_N});
  file->write("norms",
              memory::ArrayConstView<double, 1>(norms.data(),
                                                memory::Layout<1>(norms.size()),
                                                memory::MemoryType::HOST),
              {dim_M});

  return 0;
}
