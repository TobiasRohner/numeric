#include <limits>
#include <numeric/math/optim/nelder_mead.hpp>
#include <numeric/math/quad/optimize_quad_tria.hpp>
#include <numeric/math/reduce.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_op.hpp>
#include <random>

namespace numeric::math::quad {

template <typename Scalar>
static void fill_chebyshev(Scalar x, memory::ArrayView<Scalar, 1> T) {
  const dim_t order = T.shape(0) - 1;
  T(0) = 1;
  if (order > 0) {
    T(1) = x;
  }
  for (dim_t i = 1; i < order; ++i) {
    T(i + 1) = 2 * x * T(i) - T(i - 1);
  }
}

template <typename Scalar> static Scalar exact_integral(dim_t m, dim_t n) {
  const auto cint = [](dim_t i) -> Scalar {
    if (i == 1) {
      return 0;
    } else {
      const Scalar m1pi = i % 2 == 0 ? 1 : -1;
      return 0.5 * (m1pi + 1) / (1 - i * i);
    }
  };
  const auto cprod = [&cint](dim_t i, dim_t j) -> Scalar {
    return 0.5 * (cint(i + j) + cint(std::abs(i - j)));
  };
  if (n == 0) {
    return 0.5 * (cprod(m, 0) - cprod(m, 1));
  } else if (n == 1) {
    return 0.125 * (cprod(m, 2) - cprod(m, 0));
  } else {
    const Scalar mipn = n % 2 == 0 ? 1 : -1;
    const Scalar mipnm1 = n % 2 == 0 ? -1 : 1;
    return -0.5 * mipn / (n * n - 1) * cint(m) -
           0.25 * mipnm1 / (n - 1) * cprod(m, n - 1) +
           0.25 * mipnm1 / (n + 1) * cprod(m, n + 1);
  }
}

template <typename T>
std::tuple<memory::Array<T, 1>, memory::Array<T, 2>>
orbits_to_concrete(const memory::ArrayConstView<T, 1> &w0,
                   const memory::ArrayConstView<T, 2> &p0,
                   const memory::ArrayConstView<T, 1> &w1,
                   const memory::ArrayConstView<T, 2> &p1,
                   const memory::ArrayConstView<T, 1> &w2,
                   const memory::ArrayConstView<T, 2> &p2) {
  const dim_t N0 = w0.shape(0);
  const dim_t N1 = w1.shape(0);
  const dim_t N2 = w2.shape(0);
  const dim_t N = N0 + 3 * N1 + 6 * N2;

  memory::Array<T, 1> w(memory::Shape<1>(N), memory::MemoryType::HOST);
  memory::Array<T, 2> p(memory::Shape<2>(N, 2), memory::MemoryType::HOST);

  for (dim_t i = 0; i < N0; ++i) {
    w(i) = abs(w0(i));
    p(i, 0) = 1. / 3;
    p(i, 1) = 1. / 3;
  }
  for (dim_t i = 0; i < N1; ++i) {
    const T ow1 = abs(w1(i));
    const T op1 = 0.5 + 0.5 * tanh(p1(i, 0));
    w(N0 + 3 * i + 0) = ow1;
    p(N0 + 3 * i + 0, 0) = op1 / 2;
    p(N0 + 3 * i + 0, 1) = op1 / 2;
    w(N0 + 3 * i + 1) = ow1;
    p(N0 + 3 * i + 1, 0) = op1 / 2;
    p(N0 + 3 * i + 1, 1) = 1 - op1;
    w(N0 + 3 * i + 2) = ow1;
    p(N0 + 3 * i + 2, 0) = 1 - op1;
    p(N0 + 3 * i + 2, 1) = op1 / 2;
  }
  for (dim_t i = 0; i < N2; ++i) {
    const T ow2 = abs(w2(i));
    const T op2xt = 0.5 + 0.5 * tanh(p2(i, 0));
    const T op2yt = 0.5 + 0.5 * tanh(p2(i, 1));
    const T op2x = op2xt * (1 - op2yt) + 0.5 * op2xt * op2yt;
    const T op2y = (1 - op2xt) * op2yt + 0.5 * op2xt * op2yt;
    w(N0 + 3 * N1 + 6 * i + 0) = ow2;
    p(N0 + 3 * N1 + 6 * i + 0, 0) = 1. / 3 - op2x / 3 + op2y / 6;
    p(N0 + 3 * N1 + 6 * i + 0, 1) = 1. / 3 - op2x / 3 - op2y / 3;
    w(N0 + 3 * N1 + 6 * i + 1) = ow2;
    p(N0 + 3 * N1 + 6 * i + 1, 0) = 1. / 3 + 2 * op2x / 3 + op2y / 6;
    p(N0 + 3 * N1 + 6 * i + 1, 1) = 1. / 3 - op2x / 3 - op2y / 3;
    w(N0 + 3 * N1 + 6 * i + 2) = ow2;
    p(N0 + 3 * N1 + 6 * i + 2, 0) = 1. / 3 + 2 * op2x / 3 + op2y / 6;
    p(N0 + 3 * N1 + 6 * i + 2, 1) = 1. / 3 - op2x / 3 + op2y / 6;
    w(N0 + 3 * N1 + 6 * i + 3) = ow2;
    p(N0 + 3 * N1 + 6 * i + 3, 0) = 1. / 3 - op2x / 3 + op2y / 6;
    p(N0 + 3 * N1 + 6 * i + 3, 1) = 1. / 3 + 2 * op2x / 3 + op2y / 6;
    w(N0 + 3 * N1 + 6 * i + 4) = ow2;
    p(N0 + 3 * N1 + 6 * i + 4, 0) = 1. / 3 - op2x / 3 - op2y / 3;
    p(N0 + 3 * N1 + 6 * i + 4, 1) = 1. / 3 + 2 * op2x / 3 + op2y / 6;
    w(N0 + 3 * N1 + 6 * i + 5) = ow2;
    p(N0 + 3 * N1 + 6 * i + 5, 0) = 1. / 3 - op2x / 3 - op2y / 3;
    p(N0 + 3 * N1 + 6 * i + 5, 1) = 1. / 3 - op2x / 3 + op2y / 6;
  }
  w /= 2 * math::sum(w);

  return {w, p};
}

template <typename T>
static T accuracy(const memory::ArrayConstView<T, 1> &w0,
                  const memory::ArrayConstView<T, 2> &p0,
                  const memory::ArrayConstView<T, 1> &w1,
                  const memory::ArrayConstView<T, 2> &p1,
                  const memory::ArrayConstView<T, 1> &w2,
                  const memory::ArrayConstView<T, 2> &p2, dim_t order) {
  const auto [w, p] = orbits_to_concrete(w0, p0, w1, p1, w2, p2);
  const dim_t N = w.shape(0);

  // Evaluate Chebyshev Polynomials at Quadrature Points
  memory::Array<T, 2> Tx(memory::Shape<2>(N, order + 1),
                         memory::MemoryType::HOST);
  memory::Array<T, 2> Ty(memory::Shape<2>(N, order + 1),
                         memory::MemoryType::HOST);
  for (dim_t i = 0; i < N; ++i) {
    fill_chebyshev(2 * p(i, 0) - 1, Tx(i));
    fill_chebyshev(2 * p(i, 1) - 1, Ty(i));
  }

  // Compute the maximum error in any basis function
  T max_err = 0;
  for (dim_t m = 0; m <= order; ++m) {
    for (dim_t n = 0; n <= order - m; ++n) {
      const T exact = exact_integral<T>(m, n);
      T approx = 0;
      for (dim_t i = 0; i < N; ++i) {
        approx += w(i) * Tx(i, m) * Ty(i, n);
      }
      const T err = abs(approx - exact) * abs(approx - exact);
      /*
      if (err > max_err) {
        max_err = err;
      }
      */
      max_err += err;
    }
  }

  return max_err;
}

utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>
optimize_quad_tria(dim_t order, dim_t N0, dim_t N1, dim_t N2) {
  using Scalar = float;

  const Scalar tol = std::sqrt(std::numeric_limits<Scalar>::epsilon());
  static const dim_t max_iter = 10000;
  const dim_t N = N0 + 2 * N1 + 3 * N2;

  std::mt19937 rng;

  const auto x_to_orbits = [&](const memory::ArrayConstView<Scalar, 1> &x)
      -> std::tuple<
          memory::ArrayConstView<Scalar, 1>, memory::ArrayConstView<Scalar, 2>,
          memory::ArrayConstView<Scalar, 1>, memory::ArrayConstView<Scalar, 2>,
          memory::ArrayConstView<Scalar, 1>,
          memory::ArrayConstView<Scalar, 2>> {
    using sl = memory::Slice;
    const memory::ArrayConstView<Scalar, 1> w0 = x(sl(0, N0));
    const memory::ArrayConstView<Scalar, 1> w1 = x(sl(N0, N0 + N1));
    const memory::ArrayConstView<Scalar, 1> w2 = x(sl(N0 + N1, N0 + N1 + N2));
    const memory::ArrayConstView<Scalar, 2> p0(x.raw() + N0 + N1 + N2,
                                               memory::Layout<2>(N0, 0),
                                               memory::MemoryType::HOST);
    const memory::ArrayConstView<Scalar, 2> p1(x.raw() + N0 + N1 + N2,
                                               memory::Layout<2>(N1, 1),
                                               memory::MemoryType::HOST);
    const memory::ArrayConstView<Scalar, 2> p2(x.raw() + N0 + N1 + N2 + N1,
                                               memory::Layout<2>(N2, 2),
                                               memory::MemoryType::HOST);
    return {w0, p0, w1, p1, w2, p2};
  };

  const auto f = [&](const memory::ArrayConstView<Scalar, 1> &x) {
    const auto [w0, p0, w1, p1, w2, p2] = x_to_orbits(x);
    return accuracy(w0, p0, w1, p1, w2, p2, order);
  };

  const auto grad_f = [&](const memory::ArrayConstView<Scalar, 1> &x,
                          memory::ArrayView<Scalar, 1> df) {
    const Scalar dx = sqrt(std::numeric_limits<Scalar>::epsilon());
    const dim_t N = x.shape(0);
    memory::Array<Scalar, 1> x1 = x;
    const Scalar fx = f(x);
    for (dim_t i = 0; i < N; ++i) {
      x1(i) += dx;
      const Scalar fxdx = f(x1);
      x1(i) = x(i);
      df(i) = (fxdx - fx) / dx;
      // std::cout << "x(" << i << ") = " << x(i) << ", x1(" << i << ") = " <<
      // x1(i) << ", df(" << i << ") = " << df(i) << std::endl;
    }
  };

  dim_t num_iter = 0;
  const auto term_crit = [&](const memory::ArrayConstView<Scalar, 1> &x0,
                             const memory::ArrayConstView<Scalar, 2> &simplex) {
    ++num_iter;
    if (num_iter >= max_iter) {
      return true;
    }
    const dim_t N = simplex.shape(0);
    for (dim_t n = 0; n < N; ++n) {
      const Scalar dist = sqrt(math::sum(memory::pow<2>(simplex(n) - x0)));
      if (dist >= tol) {
        return false;
      }
    }
    return true;
  };

  for (dim_t k = 0; k < 10000 * N * N; ++k) {
    /*
    num_iter = 0;
    const memory::Array<Scalar, 1> x0 = memory::Array<Scalar, 1>::uniform(0, 3,
    memory::Shape<1>(N0+N1+N2+N1+2*N2), rng); const memory::Array<Scalar, 1>
    xopt = math::optim::NelderMead<Scalar>().minimize(f, x0,
    term_crit, 1./sqrt(N0+3*N1+6*N2)).x(); const Scalar fopt = f(xopt);
    std::cout << "Iteration " << (k+1) << "/" << (10000*N*N) << ", error is " <<
    fopt << '\n';
    */
    memory::Array<Scalar, 1> x0 = memory::Array<Scalar, 1>::uniform(
        0, 3, memory::Shape<1>(N0 + N1 + N2 + N1 + 2 * N2), rng);
    memory::Array<Scalar, 1> df(x0.shape(), memory::MemoryType::HOST);
    grad_f(x0, df);
    Scalar norm_df = math::sum(memory::abs(df)) / df.shape(0);
    while (norm_df >= tol) {
      x0 -= 0.00005 * df;
      grad_f(x0, df);
      norm_df = math::sum(memory::abs(df)) / df.shape(0);
      std::cout << f(x0) << ", " << norm_df << std::endl;
    }
    const Scalar fopt = f(x0);
    const auto &xopt = x0;

    if (fopt < tol) {
      const auto [w0, p0, w1, p1, w2, p2] = x_to_orbits(xopt);
      auto [w, p] = orbits_to_concrete(w0, p0, w1, p1, w2, p2);
      memory::Array<double, 1> dw = w;
      memory::Array<double, 2> dp = p;
      return utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>(
          std::move(dp), std::move(dw));
    }
  }

  return utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>(
      memory::Array<double, 2>(), memory::Array<double, 1>());
}

} // namespace numeric::math::quad
