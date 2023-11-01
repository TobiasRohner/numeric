#ifndef NUMERIC_MATH_ARRAY_OP_HPP_
#define NUMERIC_MATH_ARRAY_OP_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_base.hpp>
#include <numeric/memory/array_const_view.hpp>
#include <numeric/memory/array_traits.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::memory {

template <typename Op, typename Arg>
class ArrayUnaryOp : public ArrayBase<ArrayUnaryOp<Op, Arg>> {
  using super = ArrayBase<ArrayUnaryOp<Op, Arg>>;

public:
  using scalar_t =
      decltype(meta::declval<Op>()(meta::declval<typename Arg::scalar_t>()));
  static constexpr dim_t dim = Arg::dim;

  NUMERIC_HOST_DEVICE ArrayUnaryOp(const Op op, const Arg &arg)
      : op_(op), arg_(arg) {}
  ArrayUnaryOp(const ArrayUnaryOp &) = default;
  ArrayUnaryOp &operator=(const ArrayUnaryOp &) = default;

  NUMERIC_HOST_DEVICE [[nodiscard]] MemoryType memory_type() const {
    return arg_.memory_type();
  }

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE [[nodiscard]] const scalar_t
  operator()(Idxs... idxs) const noexcept {
    return op_(arg_(idxs...));
  }

  NUMERIC_HOST_DEVICE [[nodiscard]] dim_t shape(size_t idx) const noexcept {
    return arg_.shape(idx);
  }

  NUMERIC_HOST_DEVICE [[nodiscard]] Layout<dim> layout() const noexcept {
    return arg_.layout();
  }

  NUMERIC_HOST_DEVICE [[nodiscard]] dim_t size() const noexcept {
    return arg_.size();
  }

  template <dim_t M>
  NUMERIC_HOST_DEVICE [[nodiscard]] auto
  broadcast(const Layout<M> &layout) const noexcept {
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
class ArrayBinaryOp : public ArrayBase<ArrayBinaryOp<Op, Lhs, Rhs>> {
  using super = ArrayBase<ArrayBinaryOp<Op, Lhs, Rhs>>;

public:
  using lhs_scalar_t = typename Lhs::scalar_t;
  using rhs_scalar_t = typename Rhs::scalar_t;
  using scalar_t = decltype(meta::declval<Op>()(meta::declval<lhs_scalar_t>(),
                                                meta::declval<rhs_scalar_t>()));
  static constexpr dim_t dim = Lhs::dim > Rhs::dim ? Lhs::dim : Rhs::dim;
  using lhs_t =
      decltype(meta::declval<Lhs>().broadcast(meta::declval<Layout<dim>>()));
  using rhs_t =
      decltype(meta::declval<Rhs>().broadcast(meta::declval<Layout<dim>>()));

  NUMERIC_HOST_DEVICE ArrayBinaryOp(const Op op, const Lhs &lhs, const Rhs &rhs)
      : op_(op), lhs_(lhs), rhs_(rhs) {}
  ArrayBinaryOp(const ArrayBinaryOp &) = default;
  ArrayBinaryOp &operator=(const ArrayBinaryOp &) = default;

  NUMERIC_HOST_DEVICE [[nodiscard]] MemoryType memory_type() const {
    const MemoryType mtl = lhs_.memory_type();
    const MemoryType mtr = rhs_.memory_type();
    if (mtl == mtr) {
      return mtl;
    } else if (is_host_accessible(mtl) && is_host_accessible(mtr)) {
      return MemoryType::HOST;
#if NUMERIC_ENABLE_HIP
    } else if (is_device_accessible(mtl) && is_device_accessible(mtr)) {
      return MemoryType::DEVICE;
#endif
    } else {
      return MemoryType::UNKNOWN;
    }
  }

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE [[nodiscard]] decltype(auto)
  operator()(Idxs... idxs) const noexcept {
    if constexpr (sizeof...(idxs) == dim) {
      return op_(lhs_(idxs...), rhs_(idxs...));
    } else {
      using lhs_slice_t = meta::remove_cvref_t<decltype(lhs_(idxs...))>;
      using rhs_slice_t = meta::remove_cvref_t<decltype(rhs_(idxs...))>;
      return ArrayBinaryOp<Op, lhs_slice_t, rhs_slice_t>(op_, lhs_(idxs...),
                                                         rhs_(idxs...));
    }
  }

  NUMERIC_HOST_DEVICE [[nodiscard]] dim_t shape(size_t idx) const noexcept {
    const dim_t shape_lhs = lhs_.shape(idx);
    const dim_t shape_rhs = rhs_.shape(idx);
    return shape_lhs > shape_rhs ? shape_lhs : shape_rhs;
  }

  NUMERIC_HOST_DEVICE [[nodiscard]] Layout<dim> layout() const noexcept {
    Layout<dim> l;
    for (dim_t d = 0; d < dim; ++d) {
      l.shape(d) = shape(d);
    }
    return l;
  }

  NUMERIC_HOST_DEVICE [[nodiscard]] dim_t size() const noexcept {
    dim_t s = 1;
    for (dim_t d = 0; d < dim; ++d) {
      s *= shape(d);
    }
    return s;
  }

  template <dim_t M>
  NUMERIC_HOST_DEVICE [[nodiscard]] auto
  broadcast(const Layout<M> &layout) const noexcept {
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

template <typename Op, typename Arg> struct ArrayTraits<ArrayUnaryOp<Op, Arg>> {
  static constexpr dim_t dim = ArrayTraits<Arg>::dim;
  using scalar_t = decltype(meta::declval<Op>()(
      meta::declval<typename ArrayTraits<Arg>::scalar_t>()));
};

template <typename Op, typename Lhs, typename Rhs>
struct ArrayTraits<ArrayBinaryOp<Op, Lhs, Rhs>> {
  static constexpr dim_t lhs_dim = ArrayTraits<Lhs>::dim;
  static constexpr dim_t rhs_dim = ArrayTraits<Rhs>::dim;
  static constexpr dim_t dim = lhs_dim > rhs_dim ? lhs_dim : rhs_dim;
  using lhs_scalar_t = typename ArrayTraits<Lhs>::scalar_t;
  using rhs_scalar_t = typename ArrayTraits<Rhs>::scalar_t;
  using scalar_t = decltype(meta::declval<Op>()(meta::declval<lhs_scalar_t>(),
                                                meta::declval<rhs_scalar_t>()));
};

} // namespace numeric::memory

template <typename Arg>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::memory::ArrayUnaryOp<numeric::memory::OpUnaryPlus,
                                                Arg>
    operator+(const numeric::memory::ArrayBase<Arg> &arg) noexcept {
  return {numeric::memory::OpUnaryPlus{}, arg.derived()};
}

template <typename Arg>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::memory::ArrayUnaryOp<numeric::memory::OpUnaryMinus,
                                                Arg>
    operator-(const numeric::memory::ArrayBase<Arg> &arg) noexcept {
  return {numeric::memory::OpUnaryMinus{}, arg.derived()};
}

template <typename Arg>
NUMERIC_HOST_DEVICE [[nodiscard]] numeric::memory::ArrayUnaryOp<
    numeric::memory::OpUnaryBitwiseNot, Arg>
operator~(const numeric::memory::ArrayBase<Arg> &arg) noexcept {
  return {numeric::memory::OpUnaryBitwiseNot{}, arg.derived()};
}

template <typename Arg>
NUMERIC_HOST_DEVICE [[nodiscard]] numeric::memory::ArrayUnaryOp<
    numeric::memory::OpUnaryLogicalNot, Arg>
operator!(const numeric::memory::ArrayBase<Arg> &arg) noexcept {
  return {numeric::memory::OpUnaryLogicalNot{}, arg.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::memory::ArrayBinaryOp<numeric::memory::OpBinaryPlus,
                                                 Lhs, Rhs>
    operator+(const numeric::memory::ArrayBase<Lhs> &lhs,
              const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::memory::OpBinaryPlus{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::memory::ArrayBinaryOp<numeric::memory::OpBinaryMinus,
                                                 Lhs, Rhs>
    operator-(const numeric::memory::ArrayBase<Lhs> &lhs,
              const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::memory::OpBinaryMinus{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE [[nodiscard]] numeric::memory::ArrayBinaryOp<
    numeric::memory::OpBinaryMultiply, Lhs, Rhs>
operator*(const numeric::memory::ArrayBase<Lhs> &lhs,
          const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::memory::OpBinaryMultiply{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE [[nodiscard]] numeric::memory::ArrayBinaryOp<
    numeric::memory::OpBinaryDivide, Lhs, Rhs>
operator/(const numeric::memory::ArrayBase<Lhs> &lhs,
          const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::memory::OpBinaryDivide{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE [[nodiscard]] numeric::memory::ArrayBinaryOp<
    numeric::memory::OpBinaryModulo, Lhs, Rhs>
operator%(const numeric::memory::ArrayBase<Lhs> &lhs,
          const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::memory::OpBinaryModulo{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE [[nodiscard]] numeric::memory::ArrayBinaryOp<
    numeric::memory::OpBinaryBitwiseAnd, Lhs, Rhs>
operator&(const numeric::memory::ArrayBase<Lhs> &lhs,
          const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::memory::OpBinaryBitwiseAnd{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE [[nodiscard]] numeric::memory::ArrayBinaryOp<
    numeric::memory::OpBinaryBitwiseOr, Lhs, Rhs>
operator|(const numeric::memory::ArrayBase<Lhs> &lhs,
          const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::memory::OpBinaryBitwiseOr{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE [[nodiscard]] numeric::memory::ArrayBinaryOp<
    numeric::memory::OpBinaryBitwiseXor, Lhs, Rhs>
operator^(const numeric::memory::ArrayBase<Lhs> &lhs,
          const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::memory::OpBinaryBitwiseXor{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE [[nodiscard]] numeric::memory::ArrayBinaryOp<
    numeric::memory::OpBinaryLogicalAnd, Lhs, Rhs>
operator&&(const numeric::memory::ArrayBase<Lhs> &lhs,
           const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::memory::OpBinaryLogicalAnd{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE [[nodiscard]] numeric::memory::ArrayBinaryOp<
    numeric::memory::OpBinaryLogicalOr, Lhs, Rhs>
operator||(const numeric::memory::ArrayBase<Lhs> &lhs,
           const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::memory::OpBinaryLogicalOr{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE [[nodiscard]] numeric::memory::ArrayBinaryOp<
    numeric::memory::OpBinaryEquals, Lhs, Rhs>
operator==(const numeric::memory::ArrayBase<Lhs> &lhs,
           const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::memory::OpBinaryEquals{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE [[nodiscard]] numeric::memory::ArrayBinaryOp<
    numeric::memory::OpBinaryNotEquals, Lhs, Rhs>
operator!=(const numeric::memory::ArrayBase<Lhs> &lhs,
           const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::memory::OpBinaryNotEquals{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::memory::ArrayBinaryOp<numeric::memory::OpBinaryLess,
                                                 Lhs, Rhs>
    operator<(const numeric::memory::ArrayBase<Lhs> &lhs,
              const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::memory::OpBinaryLess{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE [[nodiscard]] numeric::memory::ArrayBinaryOp<
    numeric::memory::OpBinaryGreater, Lhs, Rhs>
operator>(const numeric::memory::ArrayBase<Lhs> &lhs,
          const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::memory::OpBinaryGreater{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::memory::ArrayBinaryOp<numeric::memory::OpBinaryLeq,
                                                 Lhs, Rhs>
    operator<=(const numeric::memory::ArrayBase<Lhs> &lhs,
               const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::memory::OpBinaryLeq{}, lhs.derived(), rhs.derived()};
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE
    [[nodiscard]] numeric::memory::ArrayBinaryOp<numeric::memory::OpBinaryGeq,
                                                 Lhs, Rhs>
    operator>=(const numeric::memory::ArrayBase<Lhs> &lhs,
               const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {
  return {numeric::memory::OpBinaryGeq{}, lhs.derived(), rhs.derived()};
}

#endif
