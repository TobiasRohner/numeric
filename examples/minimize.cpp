#include <iostream>
#include <numeric/math/optim/nelder_mead.hpp>
#include <numeric/memory/array.hpp>

double rosenbrock(double a, double b, double x, double y) {
  const double amx = a - x;
  const double ymx2 = y - x * x;
  return amx * amx + b * ymx2 * ymx2;
}

int main() {
  const double a = 1;
  const double b = 100;

  const double min_x = a;
  const double min_y = a * a;

  const auto f = [&](const numeric::memory::ArrayConstView<double, 1> &x) {
    return rosenbrock(a, b, x(0), x(1));
  };
  const auto term_crit =
      [&](const numeric::memory::ArrayConstView<double, 1> &x0,
          const numeric::memory::ArrayConstView<double, 2> &simplex) {
        const double dx = x0(0) - min_x;
        const double dy = x0(1) - min_y;
        const double err = std::sqrt(dx * dx + dy * dy);
        std::cout << "Error is " << err << " at x0 = [" << x0(0) << ", "
                  << x0(1) << "]\n";
        return err < 1e-8;
      };

  numeric::memory::Array<double, 1> x0(numeric::memory::Shape<1>(2),
                                       numeric::memory::MemoryType::HOST);
  x0(0) = 0;
  x0(1) = 0;
  numeric::math::optim::NelderMead<double>().minimize(f, x0, term_crit);

  return 0;
}
