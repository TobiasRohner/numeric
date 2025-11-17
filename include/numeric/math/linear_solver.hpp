#ifndef NUMERIC_MATH_LINEAR_SOLVER_HPP_
#define NUMERIC_MATH_LINEAR_SOLVER_HPP_

#include <memory>
#include <numeric/math/linear_operator.hpp>

namespace numeric::math {

template <typename Scalar> class LinearSolver {
public:
  using scalar_t = Scalar;
  using op_t = LinearOperator<scalar_t>;

  LinearSolver() = default;
  LinearSolver(const std::shared_ptr<op_t> &A) : A_() { set_operator(A); }
  LinearSolver(const LinearSolver &) = default;
  LinearSolver(LinearSolver &&) = default;
  virtual ~LinearSolver() = default;
  LinearSolver &operator=(const LinearSolver &) = default;
  LinearSolver &operator=(LinearSolver &&) = default;

  void set_operator(const std::shared_ptr<op_t> &A) {
    A_ = A;
    compute();
  }

  virtual void solve(const memory::ArrayConstView<scalar_t, 1> &rhs,
                     memory::ArrayView<scalar_t, 1> out) const = 0;

  memory::Array<scalar_t, 1>
  solve(const memory::ArrayConstView<scalar_t, 1> &rhs) const {
    memory::Array<scalar_t, 1> out(memory::Shape<1>(A_.shape(0)),
                                   rhs.memory_type());
    solve(rhs, out);
    return out;
  }

protected:
  std::shared_ptr<op_t> A_;

  virtual void compute() {}
};

} // namespace numeric::math

#endif
