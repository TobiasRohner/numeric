#ifndef NUMERIC_MATH_ARRAY_OP_HPP_
#define NUMERIC_MATH_ARRAY_OP_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_const_view.hpp>

namespace numeric::math {

template <typename Op, typename Arg> class ArrayUnaryOp {
public:
  using scalar_t = typename Arg::scalar_t;
  static constexpr dim_t dim = Arg::dim;

  NUMERIC_HOST_DEVICE explicit ArrayUnaryOp(const Op op, const Arg &arg)
      : op_(op), arg_(arg) {}
  NUMERIC_HOST_DEVICE ArrayUnaryOp(const ArrayUnaryOp &) = default;
  NUMERIC_HOST_DEVICE ArrayUnaryOp &operator=(const ArrayUnaryOp &) = default;

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE [[nodiscard]] const scalar_t &
  operator()(Idxs... idxs) const noexcept(noexcept(op_(scalar_t{}))) {
    return op_(arg_(idxs...));
  }

private:
  Op op_;
  Arg arg_;
};

template <typename Op, typename Lhs, typename Rhs> class ArrayBinaryOp {
public:
  using lhs_scalar_t = typename Lhs::scalar_t;
  using rhs_scalar_t = typename Rhs::scalar_t;
  using scalar_t = decltype(op_(lhs_scalar_t{}, rhs_scalar_t{}));
  static constexpr dim_t dim = Lhs::dim;
  static_assert(Lhs::dim == Rhs::dim,
                "Dimensions must match for ArrayBinaryOp");

  NUMERIC_HOST_DEVICE explicit ArrayBinaryOp(const Op op, const Lhs &lhs,
                                             const Rhs &rhs)
      : op_(op), lhs_(lhs), rhs_(rhs) {}
  NUMERIC_HOST_DEVICE ArrayBinaryOp(const ArrayBinaryOp &) = default;
  NUMERIC_HOST_DEVICE ArrayBinaryOp &operator=(const ArrayBinaryOp &) = default;

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE [[nodiscard]] const scalar_t &
  operator()(Idxs... idxs) const
      noexcept(noexcept(op_(lhs_scalar_t{}, rhs_scalar_t{}))) {
    return op_(lhs_(idxs...), rhs_(idxs...));
  }

private:
  Op op_;
  Lhs lhs_;
  Rhs rhs_;
};

} // namespace numeric::math

#endif
