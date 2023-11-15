#include <numeric/memory/array.hpp>
#include <numeric/memory/array_op.hpp>
#include <numeric/memory/linspace.hpp>
#include <numeric/memory/copyer.hpp>
#include <numeric/io/netcdf_file.hpp>
#if NUMERIC_ENABLE_HIP
#include <numeric/hip/kernel.hpp>
#endif




int main() {
  using namespace numeric;

  const double x0 = -1;
  const double x1 = 1;
  const double T = 10;
  const double c = 1;
  const dim_t N = 500;
  const double C = 0.5;
  const double dx = (x1 - x0) / (N - 1);
  const double dt = C * dx / c;
  const dim_t M = T / dt;
  const memory::MemoryType memory_type = memory::MemoryType::HOST;

  memory::Array<double, 2> u(memory::Layout<2>(M+1,N), memory_type);
  memory::Array<double, 2> v(memory::Layout<2>(M+1,N), memory_type);
  memory::Array<double, 2> a(memory::Layout<2>(M+1,N), memory_type);

  const double sigma = 0.1;
  u(0) = memory::exp(-memory::pow<2>(memory::linspace(x0, x1, N) / sigma));
  for (dim_t t = 1 ; t <= M ; ++t) {
    a(t, memory::Slice(1,N-1)) = c * c * (u(t-1,memory::Slice(2,N)) + u(t-1,memory::Slice(0,N-2)) - 2*u(t-1,memory::Slice(1,N-1))) / (dx*dx);
    v(t, memory::Slice(1,N-1)) = v(t-1, memory::Slice(1,N-1)) + dt * a(t, memory::Slice(1,N-1));
    u(t, memory::Slice(1,N-1)) = u(t-1, memory::Slice(1,N-1)) + dt * v(t, memory::Slice(1,N-1));
  }

  auto file = io::NetCDFFile::create("fd_wave.nc");
  file->write("u", u);
  //file->write("v", v);
  //file->write("a", a);

  return 0;
}
