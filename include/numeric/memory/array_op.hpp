#ifndef NUMERIC_MATH_ARRAY_OP_HPP_
#define NUMERIC_MATH_ARRAY_OP_HPP_

#include <numeric/config.hpp>
#include <numeric/math/functions.hpp>
#include <numeric/memory/array_base.hpp>
#include <numeric/memory/array_const_view.hpp>
#include <numeric/memory/array_traits.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::memory {

/**
 * @brief Class for representing a unary operation on an array.
 *
 * This class applies a unary operation to each element of an array.
 *
 * @tparam Op The unary operation functor.
 * @tparam Arg The argument array type.
 */
template <typename Op, typename Arg>
class ArrayUnaryOp : public ArrayBase<ArrayUnaryOp<Op, Arg>> {
  using super = ArrayBase<ArrayUnaryOp<Op, Arg>>;

public:
  using scalar_t =
      decltype(meta::declval<Op>()(meta::declval<typename Arg::scalar_t>()));
  static constexpr dim_t dim = Arg::dim;

  /**
   * @brief Constructs an ArrayUnaryOp object with the given unary operation and
   * argument array.
   *
   * @param op The unary operation functor.
   * @param arg The argument array.
   */
  NUMERIC_HOST_DEVICE ArrayUnaryOp(const Op op, const Arg &arg)
      : op_(op), arg_(arg) {}

  ArrayUnaryOp(const ArrayUnaryOp &) = default;
  ArrayUnaryOp &operator=(const ArrayUnaryOp &) = default;

  /**
   * @brief Returns the memory type of the array.
   *
   * @return The memory type.
   */
  NUMERIC_HOST_DEVICE MemoryType memory_type() const {
    return arg_.memory_type();
  }

  /**
   * @brief Applies the unary operation to the array.
   *
   * This function applies the unary operation to the array elements at the
   * specified indices. If the number of indices matches the dimensionality of
   * the array, the unary operation is applied directly to the element at the
   * specified indices. Otherwise, a new ArrayUnaryOp object is created with the
   * sliced array and the same unary operation.
   *
   * @tparam Idxs The types of the indices.
   * @param idxs The indices specifying the element(s) to apply the unary
   * operation to.
   * @return If the number of indices matches the dimensionality of the array,
   * returns the result of applying the unary operation to the element(s) at the
   * specified indices. Otherwise, returns a new ArrayUnaryOp object
   * representing the sliced array with the same unary operation.
   */
  template <typename... Idxs>
  NUMERIC_HOST_DEVICE decltype(auto) operator()(Idxs... idxs) const noexcept {
    if constexpr (sizeof...(Idxs) == dim) {
      return op_(arg_(idxs...));
    } else {
      using slice_t = meta::remove_cvref_t<decltype(arg_(idxs...))>;
      return ArrayUnaryOp<Op, slice_t>(op_, arg_(idxs...));
    }
  }

  /**
   * @brief Returns the shape of the array at the specified index.
   *
   * @param idx The index.
   * @return The shape at the specified index.
   */
  NUMERIC_HOST_DEVICE dim_t shape(size_t idx) const noexcept {
    return arg_.shape(idx);
  }

  /**
   * @brief Returns the shape of the array.
   *
   * @return The shape of the array.
   */
  NUMERIC_HOST_DEVICE Shape<dim> shape() const noexcept { return arg_.shape(); }

  /**
   * @brief Returns the total size of the array.
   *
   * @return The total size of the array.
   */
  NUMERIC_HOST_DEVICE dim_t size() const noexcept { return arg_.size(); }

  /**
   * @brief Broadcasts the array to the specified shape.
   *
   * @tparam M The dimensionality of the new shape.
   * @param shape The new shape to broadcast to.
   * @return The broadcasted ArrayUnaryOp object.
   */
  template <dim_t M>
  NUMERIC_HOST_DEVICE auto broadcast(const Shape<M> &shape) const noexcept {
    auto brd_arg = arg_.broadcast(shape);
    return ArrayUnaryOp<Op, decltype(brd_arg)>(op_, brd_arg);
  }

protected:
  using super::broadcasted_layout;

private:
  Op op_;
  Arg arg_;
};

/**
 * @brief Template class for performing binary operations on arrays.
 *
 * This class represents binary operations between two arrays, where the
 * operation is applied element-wise.
 *
 * @tparam Op The type of the binary operation functor.
 * @tparam Lhs The type of the left-hand side array.
 * @tparam Rhs The type of the right-hand side array.
 */
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
      decltype(meta::declval<Lhs>().broadcast(meta::declval<Shape<dim>>()));
  using rhs_t =
      decltype(meta::declval<Rhs>().broadcast(meta::declval<Shape<dim>>()));

  /**
   * @brief Constructor for ArrayBinaryOp.
   *
   * Constructs an ArrayBinaryOp object with the specified binary operation,
   * left-hand side array, and right-hand side array.
   *
   * @param op The binary operation functor.
   * @param lhs The left-hand side array.
   * @param rhs The right-hand side array.
   */
  NUMERIC_HOST_DEVICE ArrayBinaryOp(const Op op, const Lhs &lhs, const Rhs &rhs)
      : op_(op), lhs_(lhs.broadcast(shape(lhs.shape(), rhs.shape()))),
        rhs_(rhs.broadcast(shape(lhs.shape(), rhs.shape()))) {}

  ArrayBinaryOp(const ArrayBinaryOp &) = default;
  ArrayBinaryOp &operator=(const ArrayBinaryOp &) = default;

  /**
   * @brief Get the memory type of the array.
   *
   * Determines the memory type of the array based on the memory types of its
   * constituent arrays.
   *
   * @return MemoryType The memory type of the array.
   */
  NUMERIC_HOST_DEVICE MemoryType memory_type() const {
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

  /**
   * @brief Applies the unary operation to the array.
   *
   * This function applies the binary operation to the array elements at the
   * specified indices. If the number of indices matches the dimensionality of
   * the array, the binary operation is applied directly to the elements at the
   * specified indices. Otherwise, a new ArrayBinaryOp object is created with
   * the sliced arrays and the same binary operation.
   *
   * @tparam Idxs The types of the indices.
   * @param idxs The indices specifying the element(s) to apply the binary
   * operation to.
   * @return If the number of indices matches the dimensionality of the array,
   * returns the result of applying the binary operation to the element(s) at
   * the specified indices. Otherwise, returns a new ArrayBinaryOp object
   * representing the sliced array with the same binary operation.
   */
  template <typename... Idxs>
  NUMERIC_HOST_DEVICE decltype(auto) operator()(Idxs... idxs) const noexcept {
    if constexpr (sizeof...(idxs) == dim) {
      return op_(lhs_(idxs...), rhs_(idxs...));
    } else {
      using lhs_slice_t = meta::remove_cvref_t<decltype(lhs_(idxs...))>;
      using rhs_slice_t = meta::remove_cvref_t<decltype(rhs_(idxs...))>;
      return ArrayBinaryOp<Op, lhs_slice_t, rhs_slice_t>(op_, lhs_(idxs...),
                                                         rhs_(idxs...));
    }
  }

  /**
   * @brief Get the shape of the resulting array along a specified dimension.
   *
   * @param idx The dimension index.
   * @return dim_t The size of the dimension.
   */
  NUMERIC_HOST_DEVICE dim_t shape(size_t idx) const noexcept {
    const dim_t shape_lhs = lhs_.shape(idx);
    const dim_t shape_rhs = rhs_.shape(idx);
    return shape_lhs > shape_rhs ? shape_lhs : shape_rhs;
  }

  /**
   * @brief Get the shape of the resulting array.
   *
   * @return Shape<dim> The shape of the resulting array.
   */
  NUMERIC_HOST_DEVICE Shape<dim> shape() const noexcept {
    Shape<dim> s;
    for (dim_t d = 0; d < dim; ++d) {
      s[d] = shape(d);
    }
    return s;
  }

  /**
   * @brief Get the total number of elements in the resulting array.
   *
   * @return dim_t The total number of elements.
   */
  NUMERIC_HOST_DEVICE dim_t size() const noexcept {
    dim_t s = 1;
    for (dim_t d = 0; d < dim; ++d) {
      s *= shape(d);
    }
    return s;
  }

  /**
   * @brief Broadcast the resulting array to a specified shape.
   *
   * @tparam M The dimensionality of the new shape.
   * @param shape The new shape to broadcast to.
   * @return auto The broadcasted ArrayBinaryOp object.
   */
  template <dim_t M>
  NUMERIC_HOST_DEVICE auto broadcast(const Shape<M> &shape) const noexcept {
    auto brd_lhs = lhs_.broadcast(shape);
    auto brd_rhs = rhs_.broadcast(shape);
    return ArrayBinaryOp<Op, decltype(brd_lhs), decltype(brd_rhs)>(op_, brd_lhs,
                                                                   brd_rhs);
  }

protected:
  using super::broadcasted_layout;

private:
  Op op_;
  lhs_t lhs_;
  rhs_t rhs_;

  template <dim_t M1, dim_t M2>
  static NUMERIC_HOST_DEVICE Shape<(M1 > M2 ? M1 : M2)>
  shape(const Shape<M1> &s1, const Shape<M2> &s2) {
    if (M1 < M2) {
      return shape(s2, s1);
    } else {
      Shape<M1> s = s1;
      for (dim_t i = 0; i < M2; ++i) {
        if (s2[i] > s[i + M1 - M2]) {
          s[i + M1 - M2] = s2[i];
        }
      }
      return s;
    }
  }
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

template <dim_t p> struct OpUnaryPow {
  template <typename Scalar>
  NUMERIC_HOST_DEVICE auto operator()(Scalar val) const
      noexcept(noexcept(math::pow<p>(val))) {
    return math::pow<p>(val);
  }
};

struct OpUnaryAbs {
  template <typename Scalar>
  NUMERIC_HOST_DEVICE auto operator()(Scalar val) const
      noexcept(noexcept(math::abs(val))) {
    return math::abs(val);
  }
};

struct OpUnaryExp {
  template <typename Scalar>
  NUMERIC_HOST_DEVICE auto operator()(Scalar val) const
      noexcept(noexcept(math::exp(val))) {
    return math::exp(val);
  }
};

struct OpUnarySin {
  template <typename Scalar>
  NUMERIC_HOST_DEVICE auto operator()(Scalar val) const
      noexcept(noexcept(math::sin(val))) {
    return math::sin(val);
  }
};

struct OpUnaryCos {
  template <typename Scalar>
  NUMERIC_HOST_DEVICE auto operator()(Scalar val) const
      noexcept(noexcept(math::cos(val))) {
    return math::cos(val);
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
      noexcept(noexcept(lhs * rhs)) {
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
      noexcept(noexcept(lhs & rhs)) {
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
      noexcept(noexcept(lhs && rhs)) {
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

template <typename Op, typename Scalar> struct OpScalarLhs {
  Op op;
  Scalar scalar;

  template <typename Rhs>
  NUMERIC_HOST_DEVICE auto operator()(Rhs rhs) const
      noexcept(noexcept(op(scalar, rhs))) {
    return op(scalar, rhs);
  }
};

template <typename Op, typename Scalar> struct OpScalarRhs {
  Op op;
  Scalar scalar;

  template <typename Lhs>
  NUMERIC_HOST_DEVICE auto operator()(Lhs lhs) const
      noexcept(noexcept(op(lhs, scalar))) {
    return op(lhs, scalar);
  }
};

/**
 * @brief Template struct for traits of ArrayUnaryOp.
 *
 * This struct provides traits information for the ArrayUnaryOp class template.
 *
 * @tparam Op The type of the unary operation functor.
 * @tparam Arg The type of the argument array.
 */
template <typename Op, typename Arg> struct ArrayTraits<ArrayUnaryOp<Op, Arg>> {
  static constexpr bool is_array =
      true; /**< Indicates whether the type is an array. */
  static constexpr dim_t dim =
      ArrayTraits<Arg>::dim; /**< Dimensionality of the array. */
  using scalar_t = decltype(meta::declval<Op>()(
      meta::declval<typename ArrayTraits<Arg>::scalar_t>())); /**< Data type of
                                                                 the elements in
                                                                 the array. */
};

/**
 * @brief Template struct for traits of ArrayBinaryOp.
 *
 * This struct provides traits information for the ArrayBinaryOp class template.
 *
 * @tparam Op The type of the binary operation functor.
 * @tparam Lhs The type of the left-hand side array.
 * @tparam Rhs The type of the right-hand side array.
 */
template <typename Op, typename Lhs, typename Rhs>
struct ArrayTraits<ArrayBinaryOp<Op, Lhs, Rhs>> {
  static constexpr bool is_array =
      true; /**< Indicates whether the type is an array. */
  static constexpr dim_t lhs_dim = ArrayTraits<Lhs>::dim;
  static constexpr dim_t rhs_dim = ArrayTraits<Rhs>::dim;
  static constexpr dim_t dim =
      lhs_dim > rhs_dim ? lhs_dim
                        : rhs_dim; /**< Dimensionality of the array. */
  using lhs_scalar_t = typename ArrayTraits<Lhs>::scalar_t;
  using rhs_scalar_t = typename ArrayTraits<Rhs>::scalar_t;
  using scalar_t = decltype(meta::declval<Op>()(
      meta::declval<lhs_scalar_t>(),
      meta::declval<rhs_scalar_t>())); /**< Data type of the elements in the
                                          array. */
};

/**
 * @brief Function to raise each element of an array to a power.
 *
 * This function applies a unary power operation to each element of the input
 * array.
 *
 * @tparam p The power value.
 * @tparam Arg The type of the argument array.
 * @param arg The input array.
 * @return ArrayUnaryOp<OpUnaryPow<p>, Arg> The resulting array after applying
 * the unary power operation.
 */
template <dim_t p, typename Arg,
          typename = meta::enable_if_t<ArrayTraits<Arg>::is_array>>
NUMERIC_HOST_DEVICE ArrayUnaryOp<OpUnaryPow<p>, Arg>
pow(const ArrayBase<Arg> &arg) noexcept {
  return {OpUnaryPow<p>{}, arg.derived()};
}

/**
 * @brief Function to calculate the absolute value of each element in an array.
 *
 * This function applies a unary absolute value operation to each element of the
 * input array.
 *
 * @tparam Arg The type of the argument array.
 * @param arg The input array.
 * @return ArrayUnaryOp<OpUnaryAbs, Arg> The resulting array after applying the
 * unary absolute value operation.
 */
template <typename Arg,
          typename = meta::enable_if_t<ArrayTraits<Arg>::is_array>>
NUMERIC_HOST_DEVICE ArrayUnaryOp<OpUnaryAbs, Arg>
abs(const ArrayBase<Arg> &arg) noexcept {
  return {OpUnaryAbs{}, arg.derived()};
}

/**
 * @brief Function to calculate the exponential of each element in an array.
 *
 * This function applies a unary exponential operation to each element of the
 * input array.
 *
 * @tparam Arg The type of the argument array.
 * @param arg The input array.
 * @return ArrayUnaryOp<OpUnaryExp, Arg> The resulting array after applying the
 * unary exponential operation.
 */
template <typename Arg,
          typename = meta::enable_if_t<ArrayTraits<Arg>::is_array>>
NUMERIC_HOST_DEVICE ArrayUnaryOp<OpUnaryExp, Arg>
exp(const ArrayBase<Arg> &arg) noexcept {
  return {OpUnaryExp{}, arg.derived()};
}

/**
 * @brief Function to calculate the sine of each element in an array.
 *
 * This function applies a unary sine operation to each element of the
 * input array.
 *
 * @tparam Arg The type of the argument array.
 * @param arg The input array.
 * @return ArrayUnaryOp<OpUnarySin, Arg> The resulting array after applying the
 * unary sine operation.
 */
template <typename Arg,
          typename = meta::enable_if_t<ArrayTraits<Arg>::is_array>>
NUMERIC_HOST_DEVICE ArrayUnaryOp<OpUnarySin, Arg>
sin(const ArrayBase<Arg> &arg) noexcept {
  return {OpUnarySin{}, arg.derived()};
}

/**
 * @brief Function to calculate the cosine of each element in an array.
 *
 * This function applies a unary cosine operation to each element of the
 * input array.
 *
 * @tparam Arg The type of the argument array.
 * @param arg The input array.
 * @return ArrayUnaryOp<OpUnaryCos, Arg> The resulting array after applying the
 * unary cosine operation.
 */
template <typename Arg,
          typename = meta::enable_if_t<ArrayTraits<Arg>::is_array>>
NUMERIC_HOST_DEVICE ArrayUnaryOp<OpUnaryCos, Arg>
cos(const ArrayBase<Arg> &arg) noexcept {
  return {OpUnaryCos{}, arg.derived()};
}

} // namespace numeric::memory

#define NUMERIC_DECLARE_UNARY_OP(op, obj)                                      \
  template <typename Arg, typename = numeric::meta::enable_if_t<               \
                              numeric::memory::ArrayTraits<Arg>::is_array>>    \
  NUMERIC_HOST_DEVICE                                                          \
      [[nodiscard]] numeric::memory::ArrayUnaryOp<numeric::memory::obj, Arg>   \
      operator op(const numeric::memory::ArrayBase<Arg> &arg) noexcept {       \
    return {numeric::memory::obj{}, arg.derived()};                            \
  }

NUMERIC_DECLARE_UNARY_OP(+, OpUnaryPlus);
NUMERIC_DECLARE_UNARY_OP(-, OpUnaryMinus);
NUMERIC_DECLARE_UNARY_OP(~, OpUnaryBitwiseNot);
NUMERIC_DECLARE_UNARY_OP(!, OpUnaryLogicalNot);

#undef NUMERIC_DECLARE_UNARY_OP

#define NUMERIC_DECLARE_BINARY_OP(op, obj)                                     \
  template <typename Lhs, typename Rhs,                                        \
            typename = numeric::meta::enable_if_t<                             \
                numeric::memory::ArrayTraits<Lhs>::is_array &&                 \
                numeric::memory::ArrayTraits<Rhs>::is_array>>                  \
  NUMERIC_HOST_DEVICE                                                          \
      [[nodiscard]] numeric::memory::ArrayBinaryOp<numeric::memory::obj, Lhs,  \
                                                   Rhs>                        \
      operator op(const numeric::memory::ArrayBase<Lhs> &lhs,                  \
                  const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {       \
    return {numeric::memory::obj{}, lhs.derived(), rhs.derived()};             \
  }                                                                            \
                                                                               \
  template <typename Lhs, typename Rhs,                                        \
            typename = numeric::meta::enable_if_t<                             \
                !numeric::memory::ArrayTraits<Lhs>::is_array &&                \
                numeric::memory::ArrayTraits<Rhs>::is_array>>                  \
  NUMERIC_HOST_DEVICE [[nodiscard]] numeric::memory::ArrayUnaryOp<             \
      numeric::memory::OpScalarLhs<numeric::memory::obj, Lhs>, Rhs>            \
  operator op(Lhs lhs, const numeric::memory::ArrayBase<Rhs> &rhs) noexcept {  \
    return {numeric::memory::OpScalarLhs<numeric::memory::obj, Lhs>{           \
                numeric::memory::obj{}, lhs},                                  \
            rhs.derived()};                                                    \
  }                                                                            \
                                                                               \
  template <typename Lhs, typename Rhs,                                        \
            typename = numeric::meta::enable_if_t<                             \
                numeric::memory::ArrayTraits<Lhs>::is_array &&                 \
                !numeric::memory::ArrayTraits<Rhs>::is_array>>                 \
  NUMERIC_HOST_DEVICE [[nodiscard]] numeric::memory::ArrayUnaryOp<             \
      numeric::memory::OpScalarRhs<numeric::memory::obj, Rhs>, Lhs>            \
  operator op(const numeric::memory::ArrayBase<Lhs> &lhs, Rhs rhs) noexcept {  \
    return {numeric::memory::OpScalarRhs<numeric::memory::obj, Rhs>{           \
                numeric::memory::obj{}, rhs},                                  \
            lhs.derived()};                                                    \
  }

NUMERIC_DECLARE_BINARY_OP(+, OpBinaryPlus);
NUMERIC_DECLARE_BINARY_OP(-, OpBinaryMinus);
NUMERIC_DECLARE_BINARY_OP(*, OpBinaryMultiply);
NUMERIC_DECLARE_BINARY_OP(/, OpBinaryDivide);
NUMERIC_DECLARE_BINARY_OP(%, OpBinaryModulo);
NUMERIC_DECLARE_BINARY_OP(&, OpBinaryBitwiseAnd);
NUMERIC_DECLARE_BINARY_OP(|, OpBinaryBitwiseOr);
NUMERIC_DECLARE_BINARY_OP(^, OpBinaryBitwiseXor);
NUMERIC_DECLARE_BINARY_OP(&&, OpBinaryLogicalAnd);
NUMERIC_DECLARE_BINARY_OP(||, OpBinaryLogicalOr);
NUMERIC_DECLARE_BINARY_OP(==, OpBinaryEquals);
NUMERIC_DECLARE_BINARY_OP(!=, OpBinaryNotEquals);
NUMERIC_DECLARE_BINARY_OP(<, OpBinaryLess);
NUMERIC_DECLARE_BINARY_OP(>, OpBinaryGreater);
NUMERIC_DECLARE_BINARY_OP(<=, OpBinaryLeq);
NUMERIC_DECLARE_BINARY_OP(>=, OpBinaryGeq);

#undef NUMERIC_DECLARE_BINARY_OP

#endif
