#ifndef NUMERIC_MATH_CONJUGATE_GRADIENT_HPP_
#define NUMERIC_MATH_CONJUGATE_GRADIENT_HPP_

#include <numeric/math/linear_solver.hpp>
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
 *
 * @tparam Scalar Type of the scalar values (e.g., float, double).
 */
template <typename Scalar>
class ConjugateGradient : public LinearSolver<Scalar> {
  using super = LinearSolver<Scalar>;

public:
  using scalar_t = typename super::scalar_t;
  using op_t = typename super::op_t;

  /**
   * @brief Struct containing the result of the CG solver.
   */
  struct Result {
    bool converged;       ///< Whether the method converged within tolerance.
    dim_t num_iterations; ///< Number of iterations performed.
    Scalar error;         ///< Final relative residual norm.
  };

  ConjugateGradient() = default;
  /**
   * @brief Constructor that stores the matrix map A.
   * @param A Matrix-vector multiplication operator.
   */
  ConjugateGradient(const std::shared_ptr<op_t> &A) : super(A) {}
  ConjugateGradient(const ConjugateGradient &) = default;
  ConjugateGradient(ConjugateGradient &&) = default;
  virtual ~ConjugateGradient() override = default;
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

  const Result &result() const { return result_; }

  /**
   * @brief Solves the linear system Ax = b using the CG algorithm.
   *
   * @param b Right-hand side vector.
   * @param x In/out vector for the solution. Should be initialized to an
   * initial guess.
   */
  virtual void solve(const memory::ArrayConstView<Scalar, 1> &b,
                     memory::ArrayView<Scalar, 1> x) const override {
    // Compute squared norm of b for relative error comparison
    const Scalar b2 = norm::l2_squared(b);
    const Scalar threshold = tol_ * tol_ * b2;

    // Temporary storage for A*p
    memory::Array<Scalar, 1> Ap(x.shape(), x.memory_type());

    // Initial residual r = b - A * x
    (*A_)(x, Ap);
    memory::Array<Scalar, 1> r = b - Ap;
    Scalar r2 = norm::l2_squared(r);

    // Early convergence check
    if (r2 < threshold) {
      result_.converged = true;
      result_.num_iterations = 0;
      result_.error = std::sqrt(r2 / b2);
      return;
    }

    // Initialize search direction
    memory::Array<Scalar, 1> p = r;

    for (dim_t i = 0; i < max_iters_; ++i) {
      (*A_)(p, Ap);                          // Compute A*p
      const Scalar alpha = r2 / sum(p * Ap); // Step size
      x += alpha * p;                        // Update solution
      r -= alpha * Ap;                       // Update residual

      // Compute new residual norm
      const Scalar rp12 = norm::l2_squared(r);
      if (rp12 < threshold) {
        result_.converged = true;
        result_.num_iterations = i + 1;
        result_.error = std::sqrt(rp12 / b2);
        return;
      }

      // Update search direction
      const Scalar beta = rp12 / r2;
      p = r + beta * p;
      r2 = rp12;
    }

    // Return result if not converged
    result_.converged = false;
    result_.num_iterations = max_iters_;
    result_.error = std::sqrt(r2 / b2);
  }

  using super::solve;

protected:
  using super::A_;

private:
  Scalar tol_ = 1e-8;    ///< Convergence tolerance
  dim_t max_iters_ = -1; ///< Maximum number of iterations (-1 = unlimited)
  mutable Result result_;
};

} // namespace numeric::math

#endif
