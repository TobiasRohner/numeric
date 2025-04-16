#ifndef NUMERIC_MATH_CONJUGATE_GRADIENT_HPP_
#define NUMERIC_MATH_CONJUGATE_GRADIENT_HPP_

#include <numeric/math/norm.hpp>
#include <numeric/math/sum.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_view.hpp>

namespace numeric::math {

/**
 * @brief Conjugate Gradient (CG) solver for symmetric positive-definite linear
 * systems.
 *
 * This class solves linear systems of the form Ax = b using the CG method.
 * The matrix A is provided as a callable map object (e.g., function or
 * functor).
 *
 * @tparam Scalar Type of the scalar values (e.g., float, double).
 * @tparam Map Type of matrix-vector multiplication operator (e.g., lambda or
 * class with `operator()`).
 */
template <typename Scalar, typename Map> class ConjugateGradient {
public:
  /**
   * @brief Struct containing the result of the CG solver.
   */
  struct Result {
    bool converged;       ///< Whether the method converged within tolerance.
    dim_t num_iterations; ///< Number of iterations performed.
    Scalar error;         ///< Final relative residual norm.
  };

  /**
   * @brief Constructor that stores the matrix map A.
   * @param A Matrix-vector multiplication operator.
   */
  ConjugateGradient(const Map &A) : A_(A) {}
  ConjugateGradient(const ConjugateGradient &) = default;
  ConjugateGradient(ConjugateGradient &&) = default;
  ConjugateGradient &operator=(const ConjugateGradient &) = default;
  ConjugateGradient &operator=(ConjugateGradient &&) = default;

  /// Get current convergence tolerance.
  Scalar tolerance() const { return tol_; }

  /// Set convergence tolerance.
  void set_tolerance(Scalar val) { tol_ = val; }

  /// Get maximum number of allowed iterations.
  dim_t max_iterations() const { return max_iters_; }

  /// Set maximum number of allowed iterations.
  void set_max_iterations(dim_t val) { max_iters_ = val; }

  /**
   * @brief Solves the linear system Ax = b using the CG algorithm.
   *
   * @tparam dim Number of dimensions of the input arrays.
   * @param b Right-hand side vector.
   * @param x In/out vector for the solution. Should be initialized to an
   * initial guess.
   * @return Result object containing convergence info.
   */
  template <dim_t dim>
  Result solve(const memory::ArrayConstView<Scalar, dim> &b,
               memory::ArrayView<Scalar, dim> x) {
    // Compute squared norm of b for relative error comparison
    const Scalar b2 = norm::l2_squared(b);
    const Scalar threshold = tol_ * tol_ * b2;

    // Initial residual r = b - A * x
    memory::Array<Scalar, dim> r = b - A_(x);
    Scalar r2 = norm::l2_squared(r);

    // Early convergence check
    if (r2 < threshold) {
      return {true, 0, std::sqrt(r2 / b2)};
    }

    // Initialize search direction and temp vector
    memory::Array<Scalar, dim> p = r;
    memory::Array<Scalar, dim> Ap(x.shape(), x.memory_type());

    for (dim_t i = 0; i < max_iters_; ++i) {
      Ap = A_(p);                            // Compute A*p
      const Scalar alpha = r2 / sum(p * Ap); // Step size
      x += alpha * p;                        // Update solution
      r -= alpha * Ap;                       // Update residual

      // Compute new residual norm
      const Scalar rp12 = norm::l2_squared(r);
      if (rp12 < threshold) {
        return {true, i, std::sqrt(rp12 / b2)};
      }

      // Update search direction
      const Scalar beta = rp12 / r2;
      p = r + beta * p;
      r2 = rp12;
    }

    // Return result if not converged
    return {false, max_iters_, std::sqrt(r2 / b2)};
  }

private:
  Map A_;                ///< Matrix-vector multiplication operator
  Scalar tol_ = 1e-8;    ///< Convergence tolerance
  dim_t max_iters_ = -1; ///< Maximum number of iterations (-1 = unlimited)
};

} // namespace numeric::math

#endif
