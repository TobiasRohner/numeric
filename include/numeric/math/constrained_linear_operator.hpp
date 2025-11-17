#ifndef NUMERIC_MATH_CONSTRAINED_LINEAR_OPERATOR_HPP_
#define NUMERIC_MATH_CONSTRAINED_LINEAR_OPERATOR_HPP_

#include <numeric/math/linear_operator.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_const_view.hpp>
#include <numeric/memory/array_view.hpp>

namespace numeric::math {

template <typename Scalar>
class ConstrainedLinearOperator : public LinearOperator<Scalar> {
  using super = LinearOperator<Scalar>;

public:
  using scalar_t = Scalar;
  using op_t = LinearOperator<scalar_t>;

  ConstrainedLinearOperator(
      const std::shared_ptr<op_t> &A,
      const memory::ArrayConstView<dim_t, 1> &constrained_dofs)
      : A_(A), constrained_dofs_(constrained_dofs), work_(A->shape(0)) {}
  ConstrainedLinearOperator(const ConstrainedLinearOperator &) = default;
  ConstrainedLinearOperator(ConstrainedLinearOperator &&) = default;
  virtual ~ConstrainedLinearOperator() override = default;
  ConstrainedLinearOperator &
  operator=(const ConstrainedLinearOperator &) = default;
  ConstrainedLinearOperator &operator=(ConstrainedLinearOperator &&) = default;

  virtual memory::Shape<2> shape() const override { return A_->shape(); }

  virtual dim_t shape(dim_t i) const override { return A_->shape(i); }

  virtual void operator()(const memory::ArrayConstView<scalar_t, 1> &u,
                          memory::ArrayView<scalar_t, 1> out) const override {
    work_ = u;
    clear_constrained(work_);
    A_->operator()(work_, out);
    copy_constrained(out, u);
  }

  virtual memory::Array<scalar_t, 1>
  operator()(const memory::ArrayConstView<scalar_t, 1> &u) const override {
    memory::Array<scalar_t, 1> out(shape(0));
    this->operator()(u, out);
    return out;
  }

  void clear_constrained(memory::ArrayView<scalar_t, 1> &x) const {
    for (dim_t i = 0; i < constrained_dofs_.size(); ++i) {
      x(constrained_dofs_(i)) = 0;
    }
  }

  void copy_constrained(memory::ArrayView<scalar_t, 1> &x,
                        const memory::ArrayConstView<scalar_t, 1> &vals) const {
    for (dim_t i = 0; i < constrained_dofs_.size(); ++i) {
      x(constrained_dofs_(i)) = vals(constrained_dofs_(i));
    }
  }

private:
  std::shared_ptr<op_t> A_;
  memory::Array<dim_t, 1> constrained_dofs_;
  mutable memory::Array<scalar_t, 1> work_;
};

} // namespace numeric::math

#endif
