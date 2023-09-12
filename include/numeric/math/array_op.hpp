#ifndef NUMERIC_MATH_ARRAY_OP_HPP_
#define NUMERIC_MATH_ARRAY_OP_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_base.hpp>
#include <numeric/memory/array_const_view.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::math {

template <typename Op, typename Arg>
class ArrayUnaryOp : public memory::ArrayBase<ArrayUnaryOp<Op, Arg>> {
  using super = memory::ArrayBase<ArrayUnaryOp<Op, Arg>>;

public:
  using scalar_t =
      decltype(meta::declval<Op>()(meta::declval<typename Arg::scalar_t>()));
  static constexpr dim_t dim = Arg::dim;

  NUMERIC_HOST_DEVICE ArrayUnaryOp(const Op op, const Arg &arg)
      : op_(op), arg_(arg) {}
  NUMERIC_HOST_DEVICE ArrayUnaryOp(const ArrayUnaryOp &) = default;
  NUMERIC_HOST_DEVICE ArrayUnaryOp &operator=(const ArrayUnaryOp &) = default;

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE [[nodiscard]] const scalar_t
  operator()(Idxs... idxs) const noexcept {
    return op_(arg_(idxs...));
  }

  NUMERIC_HOST_DEVICE [[nodiscard]] dim_t shape(size_t idx) const noexcept {
    return arg_.shape(idx);
  }

  NUMERIC_HOST_DEVICE [[nodiscard]] memory::Layout<dim>
  layout() const noexcept {
    return arg_.layout();
  }

  template <dim_t M>
  NUMERIC_HOST_DEVICE [[nodiscard]] auto
  broadcast(const memory::Layout<M> &layout) const noexcept {
    auto brd_arg = arg_.broadcast(layout);
    return ArrayUnaryOp<Op, decltype(brd_arg)>(op_, brd_arg);
  }

protected:
  using super::broadcasted_layout;

private:
  Op op_;
  Arg arg_;
};

template <typename Op, typename Lhs, typename Rhs>
class ArrayBinaryOp : public memory::ArrayBase<ArrayBinaryOp<Op, Lhs, Rhs>> {
  using super = memory::ArrayBase<ArrayBinaryOp<Op, Lhs, Rhs>>;

public:
  using lhs_scalar_t = typename Lhs::scalar_t;
  using rhs_scalar_t = typename Rhs::scalar_t;
  using scalar_t = decltype(meta::declval<Op>()(meta::declval<lhs_scalar_t>(),
                                                meta::declval<rhs_scalar_t>()));
  static constexpr dim_t dim = Lhs::dim > Rhs::dim ? Lhs::dim : Rhs::dim;
  using lhs_t = decltype(meta::declval<Lhs>().broadcast(
      meta::declval<memory::Layout<dim>>()));
  using rhs_t = decltype(meta::declval<Rhs>().broadcast(
      meta::declval<memory::Layout<dim>>()));

  NUMERIC_HOST_DEVICE ArrayBinaryOp(const Op op, const Lhs &lhs, const Rhs &rhs)
      : op_(op), lhs_(lhs), rhs_(rhs) {}
  NUMERIC_HOST_DEVICE ArrayBinaryOp(const ArrayBinaryOp &) = default;
  NUMERIC_HOST_DEVICE ArrayBinaryOp &operator=(const ArrayBinaryOp &) = default;

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE [[nodiscard]] const scalar_t
  operator()(Idxs... idxs) const noexcept {
    return op_(lhs_(idxs...), rhs_(idxs...));
  }

  NUMERIC_HOST_DEVICE [[nodiscard]] dim_t shape(size_t idx) const noexcept {
    const dim_t shape_lhs = lhs_.shape(idx);
    const dim_t shape_rhs = rhs_.shape(idx);
    return shape_lhs > shape_rhs ? shape_lhs : shape_rhs;
  }

  template <dim_t M>
  NUMERIC_HOST_DEVICE [[nodiscard]] auto
  broadcast(const memory::Layout<M> &layout) const noexcept {
    auto brd_lhs = lhs_.broadcast(layout);
    auto brd_rhs = rhs_.broadcast(layout);
    return ArrayBinaryOp<Op, decltype(brd_lhs), decltype(brd_rhs)>(op_, brd_lhs,
                                                                   brd_rhs);
  }

protected:
  using super::broadcasted_layout;

private:
  Op op_;
  lhs_t lhs_;
  rhs_t rhs_;
};

struct OpUnaryPlus {
  template <typename Scalar>
  NUMERIC_HOST_DEVICE auto operator()(Scalar val) const
      noexcept(noexcept(+val)) {
    return +val;
  }
};

struct OpUnaryMinus {
  template <typename Scalar>
  NUMERIC_HOST_DEVICE auto operator()(Scalar val) const
      noexcept(noexcept(-val)) {
    return -val;
  }
};

struct OpUnaryBitwiseNot {
  template <typename Scalar>
  NUMERIC_HOST_DEVICE auto operator()(Scalar val) const
      noexcept(noexcept(~val)) {
    return ~val;
  }
};

struct OpUnaryLogicalNot {
  template <typename Scalar>
  NUMERIC_HOST_DEVICE auto operator()(Scalar val) const
      noexcept(noexcept(!val)) {
    return !val;
  }
};

struct OpBinaryPlus {
  template <typename Lhs, typename Rhs>
  NUMERIC_HOST_DEVICE auto operator()(Lhs lhs, Rhs rhs) const
      noexcept(noexcept(lhs + rhs)) {
    return lhs + rhs;
  }
};

struct OpBinaryMinus {
  template <typename Lhs, typename Rhs>
  NUMERIC_HOST_DEVICE auto operator()(Lhs lhs, Rhs rhs) const
      noexcept(noexcept(lhs - rhs)) {
    return lhs - rhs;
  }
};

struct OpBinaryMultiply {
  template <typename Lhs, typename Rhs>
  NUMERIC_HOST_DEVICE auto operator()(Lhs lhs, Rhs rhs) const
      noexcept(noexcept(lhs *rhs)) {
    return lhs * rhs;
  }
};

struct OpBinaryDivide {
  template <typename Lhs, typename Rhs>
  NUMERIC_HOST_DEVICE auto operator()(Lhs lhs, Rhs rhs) const
      noexcept(noexcept(lhs / rhs)) {
    return lhs / rhs;
  }
};

struct OpBinaryModulo {
  template <typename Lhs, typename Rhs>
  NUMERIC_HOST_DEVICE auto operator()(Lhs lhs, Rhs rhs) const
      noexcept(noexcept(lhs % rhs)) {
    return lhs % rhs;
  }
};

struct OpBinaryBitwiseAnd {
  template <typename Lhs, typename Rhs>
  NUMERIC_HOST_DEVICE auto operator()(Lhs lhs, Rhs rhs) const
      noexcept(noexcept(lhs &rhs)) {
    return lhs & rhs;
  }
};

struct OpBinaryBitwiseOr {
  template <typename Lhs, typename Rhs>
  NUMERIC_HOST_DEVICE auto operator()(Lhs lhs, Rhs rhs) const
      noexcept(noexcept(lhs | rhs)) {
    return lhs | rhs;
  }
};

struct OpBinaryBitwiseXor {
  template <typename Lhs, typename Rhs>
  NUMERIC_HOST_DEVICE auto operator()(Lhs lhs, Rhs rhs) const
      noexcept(noexcept(lhs ^ rhs)) {
    return lhs ^ rhs;
  }
};

struct OpBinaryLogicalAnd {
  template <typename Lhs, typename Rhs>
  NUMERIC_HOST_DEVICE auto operator()(Lhs lhs, Rhs rhs) const
      noexcept(noexcept(lhs &&rhs)) {
    return lhs && rhs;
  }
};

struct OpBinaryLogicalOr {
  template <typename Lhs, typename Rhs>
  NUMERIC_HOST_DEVICE auto operator()(Lhs lhs, Rhs rhs) const
      noexcept(noexcept(lhs || rhs)) {
    return lhs || rhs;
  }
};

struct OpBinaryEquals {
  template <typename Lhs, typename Rhs>
  NUMERIC_HOST_DEVICE auto operator()(Lhs lhs, Rhs rhs) const
      noexcept(noexcept(lhs == rhs)) {
    return lhs == rhs;
  }
};

struct OpBinaryNotEquals {
  template <typename Lhs, typename Rhs>
  NUMERIC_HOST_DEVICE auto operator()(Lhs lhs, Rhs rhs) const
      noexcept(noexcept(lhs != rhs)) {
    return lhs != rhs;
  }
};

struct OpBinaryLess {
  template <typename Lhs, typename Rhs>
  NUMERIC_HOST_DEVICE auto operator()(Lhs lhs, Rhs rhs) const
      noexcept(noexcept(lhs < rhs)) {
    return lhs < rhs;
  }
};

struct OpBinaryGreater {
  template <typename Lhs, typename Rhs>
  NUMERIC_HOST_DEVICE auto operator()(Lhs lhs, Rhs rhs) const
      noexcept(noexcept(lhs > rhs)) {
    return lhs > rhs;
  }
};

struct OpBinaryLeq {
  template <typename Lhs, typename Rhs>
  NUMERIC_HOST_DEVICE auto operator()(Lhs lhs, Rhs rhs) const
      noexcept(noexcept(lhs <= rhs)) {
    return lhs <= rhs;
  }
};

struct OpBinaryGeq {
  template <typename Lhs, typename Rhs>
  NUMERIC_HOST_DEVICE auto operator()(Lhs lhs, Rhs rhs) const
      noexcept(noexcept(lhs >= rhs)) {
    return lhs >= rhs;
  }
};

} // namespace numeric::math

template <typename Arg>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::math::ArrayUnaryOp<numeric::math::OpUnaryPlus, Arg>
    operator+(const numeric::memory::ArrayBase<Arg> &arg) noexcept {
  return {numeric::math::OpUnaryPlus{}, arg.derived()};
}

template <typename Arg>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::math::ArrayUnaryOp<numeric::math::OpUnaryMinus, Arg>
    operator-(const numeric::memory::ArrayBase<Arg> &arg) noexcept {
  return {numeric::math::OpUnaryMinus{}, arg.derived()};
}

template <typename Arg>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::math::ArrayUnaryOp<numeric::math::OpUnaryBitwiseNot,
                                              Arg>
    operator~(const numeric::memory::ArrayBase<Arg> &arg) noexcept {
  return {numeric::math::OpUnaryBitwiseNot{}, arg.derived()};
}

template <typename Arg>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::math::ArrayUnaryOp<numeric::math::OpUnaryLogicalNot,
                                              Arg>
    operator!(const numeric::memory::ArrayBase<Arg> &arg) noexcept {
  return {numeric::math::OpUnaryLogicalNot{}, arg.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::math::ArrayBinaryOp<numeric::math::OpBinaryPlus, Lhs,
                                               Rhs>
    operator+(const numeric::memory::ArrayBase<Lhs> &lhs,
              const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::math::OpBinaryPlus{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::math::ArrayBinaryOp<numeric::math::OpBinaryMinus,
                                               Lhs, Rhs>
    operator-(const numeric::memory::ArrayBase<Lhs> &lhs,
              const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::math::OpBinaryMinus{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::math::ArrayBinaryOp<numeric::math::OpBinaryMultiply,
                                               Lhs, Rhs>
    operator*(const numeric::memory::ArrayBase<Lhs> &lhs,
              const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::math::OpBinaryMultiply{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::math::ArrayBinaryOp<numeric::math::OpBinaryDivide,
                                               Lhs, Rhs>
    operator/(const numeric::memory::ArrayBase<Lhs> &lhs,
              const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::math::OpBinaryDivide{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::math::ArrayBinaryOp<numeric::math::OpBinaryModulo,
                                               Lhs, Rhs>
    operator%(const numeric::memory::ArrayBase<Lhs> &lhs,
              const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::math::OpBinaryModulo{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE [[nodiscard]] numeric::math::ArrayBinaryOp<
    numeric::math::OpBinaryBitwiseAnd, Lhs, Rhs>
operator&(const numeric::memory::ArrayBase<Lhs> &lhs,
          const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::math::OpBinaryBitwiseAnd{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::math::ArrayBinaryOp<numeric::math::OpBinaryBitwiseOr,
                                               Lhs, Rhs>
    operator|(const numeric::memory::ArrayBase<Lhs> &lhs,
              const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::math::OpBinaryBitwiseOr{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE [[nodiscard]] numeric::math::ArrayBinaryOp<
    numeric::math::OpBinaryBitwiseXor, Lhs, Rhs>
operator^(const numeric::memory::ArrayBase<Lhs> &lhs,
          const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::math::OpBinaryBitwiseXor{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE [[nodiscard]] numeric::math::ArrayBinaryOp<
    numeric::math::OpBinaryLogicalAnd, Lhs, Rhs>
operator&&(const numeric::memory::ArrayBase<Lhs> &lhs,
           const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::math::OpBinaryLogicalAnd{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::math::ArrayBinaryOp<numeric::math::OpBinaryLogicalOr,
                                               Lhs, Rhs>
    operator||(const numeric::memory::ArrayBase<Lhs> &lhs,
               const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::math::OpBinaryLogicalOr{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::math::ArrayBinaryOp<numeric::math::OpBinaryEquals,
                                               Lhs, Rhs>
    operator==(const numeric::memory::ArrayBase<Lhs> &lhs,
               const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::math::OpBinaryEquals{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::math::ArrayBinaryOp<numeric::math::OpBinaryNotEquals,
                                               Lhs, Rhs>
    operator!=(const numeric::memory::ArrayBase<Lhs> &lhs,
               const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::math::OpBinaryNotEquals{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::math::ArrayBinaryOp<numeric::math::OpBinaryLess, Lhs,
                                               Rhs>
    operator<(const numeric::memory::ArrayBase<Lhs> &lhs,
              const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::math::OpBinaryLess{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::math::ArrayBinaryOp<numeric::math::OpBinaryGreater,
                                               Lhs, Rhs>
    operator>(const numeric::memory::ArrayBase<Lhs> &lhs,
              const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::math::OpBinaryGreater{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::math::ArrayBinaryOp<numeric::math::OpBinaryLeq, Lhs,
                                               Rhs>
    operator<=(const numeric::memory::ArrayBase<Lhs> &lhs,
               const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::math::OpBinaryLeq{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::math::ArrayBinaryOp<numeric::math::OpBinaryGeq, Lhs,
                                               Rhs>
    operator>=(const numeric::memory::ArrayBase<Lhs> &lhs,
               const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::math::OpBinaryGeq{}, lhs.derived(), rhs.derived()};
}

#endif
