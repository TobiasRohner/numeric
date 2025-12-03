#ifndef NUMERIC_MATH_CUSTOM_LINEAR_OPERATOR_HPP_
#define NUMERIC_MATH_CUSTOM_LINEAR_OPERATOR_HPP_

#include <numeric/math/linear_operator.hpp>

namespace numeric::math {

template <typename Scalar>
class CustomLinearOperator : public LinearOperator<Scalar> {
  using super = LinearOperator<Scalar>;

public:
  using scalar_t = typename super::scalar_t;
  using func_t = std::function<void(const memory::ArrayConstView<scalar_t, 1> &,
                                    memory::ArrayView<scalar_t, 1>)>;

  CustomLinearOperator(const memory::Shape<2> &shape, const func_t &f)
      : shape_(shape), f_(f) {}
  CustomLinearOperator(const CustomLinearOperator &) = default;
  CustomLinearOperator(CustomLinearOperator &&) = default;
  virtual ~CustomLinearOperator() override = default;
  CustomLinearOperator &operator=(const CustomLinearOperator &) = default;
  CustomLinearOperator &operator=(CustomLinearOperator &&) = default;

  virtual memory::MemoryType memory_type() const override {
    return memory::MemoryType::UNKNOWN;
  }

  virtual memory::Shape<2> shape() const override { return shape_; }

  virtual void operator()(const memory::ArrayConstView<scalar_t, 1> &x,
                          memory::ArrayView<scalar_t, 1> out) const {
    f_(x, out);
  }

private:
  memory::Shape<2> shape_;
  func_t f_;
};

} // namespace numeric::math

#endif
