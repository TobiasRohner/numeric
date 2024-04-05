#ifndef NUMERIC_MATH_CONJUGATE_GRADIENT_HPP_
#define NUMERIC_MATH_CONJUGATE_GRADIENT_HPP_

#include <numeric/math/norm.hpp>
#include <numeric/math/sum.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_view.hpp>

namespace numeric::math {

template <typename Scalar, typename Map> class ConjugateGradient {
public:
  struct Result {
    bool converged;
    dim_t num_iterations;
    Scalar error;
  };

  ConjugateGradient(const Map &A) : A_(A) {}
  ConjugateGradient(const ConjugateGradient &) = default;
  ConjugateGradient(ConjugateGradient &&) = default;
  ConjugateGradient &operator=(const ConjugateGradient &) = default;
  ConjugateGradient &operator=(ConjugateGradient &&) = default;

  Scalar tolerance() const { return tol_; }
  void set_tolerance(Scalar val) { tol_ = val; }
  dim_t max_iterations() const { return max_iters_; }
  void set_max_iterations(dim_t val) { max_iters_ = val; }

  template <dim_t dim>
  Result solve(const memory::ArrayConstView<Scalar, dim> &b,
               memory::ArrayView<Scalar, dim> x) {
    const Scalar b2 = norm::l2_squared(b);
    const Scalar threshold = tol_ * tol_ * b2;
    memory::Array<Scalar, dim> r = b - A_(x);
    Scalar r2 = norm::l2_squared(r);
    if (r2 < threshold) {
      return {true, 0, std::sqrt(r2 / b2)};
    }
    memory::Array<Scalar, dim> p = r;
    memory::Array<Scalar, dim> Ap(x.shape(), x.memory_type());
    for (dim_t i = 0; i < max_iters_; ++i) {
      Ap = A_(p);
      const Scalar alpha = r2 / sum(p * Ap);
      x += alpha * p;
      r -= alpha * Ap;
      const Scalar rp12 = norm::l2_squared(r);
      if (rp12 < threshold) {
        return {true, i, std::sqrt(rp12 / b2)};
      }
      const Scalar beta = rp12 / r2;
      p = r + beta * p;
      r2 = rp12;
    }
    return {false, max_iters_, std::sqrt(r2 / b2)};
  }

private:
  Map A_;
  Scalar tol_ = 1e-8;
  dim_t max_iters_ = -1;
};

} // namespace numeric::math

#endif
