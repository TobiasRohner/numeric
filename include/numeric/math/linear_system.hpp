#ifndef NUMERIC_MATH_LINEAR_SYSTEM_HPP_
#define NUMERIC_MATH_LINEAR_SYSTEM_HPP_

#include <memory>
#include <numeric/math/constrained_linear_operator.hpp>
#include <numeric/math/linear_operator.hpp>
#include <numeric/math/linear_solver.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_const_view.hpp>
#include <numeric/memory/array_view.hpp>

namespace numeric::math {

template <typename Scalar> class LinearSystem {
public:
  using scalar_t = Scalar;
  using op_t = LinearOperator<scalar_t>;
  using solver_t = LinearSolver<scalar_t>;

  LinearSystem(const std::shared_ptr<op_t> &A) : A_(A), rhs_(A->shape(0)) {}
  LinearSystem(const LinearSystem &) = default;
  LinearSystem(LinearSystem &&) = default;
  ~LinearSystem() = default;
  LinearSystem &operator=(const LinearSystem &) = default;
  LinearSystem &operator=(LinearSystem &&) = default;

  memory::ArrayConstView<scalar_t, 1> rhs() const { return rhs_; }

  memory::ArrayView<scalar_t, 1> rhs() { return rhs_; }

  void set_fixed_dofs(const memory::ArrayConstView<dim_t, 1> &fixed_dofs_) {
    A_constrained_ =
        std::make_shared<ConstrainedLinearOperator<scalar_t>>(A_, fixed_dofs_);
    if (solver_) {
      solver_->set_operator(A_constrained_);
    }
  }

  void set_solver(const std::shared_ptr<solver_t> &solver) {
    solver_ = solver;
    if (A_constrained_) {
      solver_->set_operator(A_constrained_);
    } else {
      solver_->set_operator(A_);
    }
  }

  void solve(memory::ArrayView<scalar_t, 1> out) {
    out = 0;
    if (A_constrained_) {
      A_constrained_->clear_constrained(rhs_);
    }
    solver_->solve(rhs_, out);
  }

  void solve(memory::ArrayView<scalar_t, 1> out,
             const memory::ArrayConstView<scalar_t, 1> &offset) {
    (*A_)(offset, out);
    rhs_ -= out;
    solve(out);
    out += offset;
  }

private:
  std::shared_ptr<op_t> A_;
  std::shared_ptr<ConstrainedLinearOperator<scalar_t>> A_constrained_;
  std::shared_ptr<solver_t> solver_;
  memory::Array<scalar_t, 1> rhs_;
};

} // namespace numeric::math

#endif
