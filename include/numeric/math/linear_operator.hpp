#ifndef NUMERIC_MATH_LINEAR_OPERATOR_HPP_
#define NUMERIC_MATH_LINEAR_OPERATOR_HPP_

#include <numeric/memory/array.hpp>
#include <numeric/memory/array_const_view.hpp>
#include <numeric/memory/array_view.hpp>
#include <numeric/memory/shape.hpp>

namespace numeric::math {

template <typename Scalar> class LinearOperator {
public:
  using scalar_t = Scalar;

  LinearOperator() = default;
  LinearOperator(const LinearOperator &) = default;
  LinearOperator(LinearOperator &&) = default;
  virtual ~LinearOperator() = default;
  LinearOperator &operator=(const LinearOperator &) = default;
  LinearOperator &operator=(LinearOperator &&) = default;

  virtual memory::Shape<2> shape() const = 0;

  virtual dim_t shape(dim_t i) const { return shape()[i]; }

  virtual void operator()(const memory::ArrayConstView<scalar_t, 1> &x,
                          memory::ArrayView<scalar_t, 1> out) const = 0;

  virtual memory::Array<scalar_t, 1>
  operator()(const memory::ArrayConstView<scalar_t, 1> &x) const {
    memory::Array<scalar_t, 1> out(memory::Shape<1>(shape(0)), x.memory_type());
    this->operator()(x, out);
    return out;
  }
};

} // namespace numeric::math

#endif
