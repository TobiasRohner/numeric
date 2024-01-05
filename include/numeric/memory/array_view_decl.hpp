#ifndef NUMERIC_MEMORY_ARRAY_VIEW_DECL_HPP_
#define NUMERIC_MEMORY_ARRAY_VIEW_DECL_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_const_view_decl.hpp>
#include <numeric/memory/array_traits.hpp>
#if NUMERIC_ENABLE_EIGEN
#include <Eigen/Dense>
#endif

namespace numeric::memory {

template <typename Scalar, dim_t N>
class ArrayView : public ArrayConstView<Scalar, N> {
  using super = ArrayConstView<Scalar, N>;

public:
  using scalar_t = Scalar;
  static constexpr dim_t dim = N;

  NUMERIC_HOST_DEVICE ArrayView();
  NUMERIC_HOST_DEVICE ArrayView(scalar_t *data, const Layout<dim> &layout,
                                MemoryType memory_type = MemoryType::UNKNOWN);
  ArrayView(const ArrayView &) = default;
  ArrayView(ArrayView &&) = default;
  NUMERIC_HOST_DEVICE ArrayView &operator=(const ArrayView &other);
  template <typename Src>
  NUMERIC_HOST_DEVICE ArrayView &operator=(const ArrayBase<Src> &src);
  NUMERIC_HOST_DEVICE ArrayView &operator=(Scalar val);
#define NUMERIC_ARRAY_VIEW_DECLARE_ASSIGNMENT(op)                              \
  template <typename Src>                                                      \
  NUMERIC_HOST_DEVICE ArrayView &operator op(const ArrayBase<Src> &src);       \
  NUMERIC_HOST_DEVICE ArrayView &operator op(Scalar val);
  NUMERIC_ARRAY_VIEW_DECLARE_ASSIGNMENT(+=);
  NUMERIC_ARRAY_VIEW_DECLARE_ASSIGNMENT(-=);
  NUMERIC_ARRAY_VIEW_DECLARE_ASSIGNMENT(*=);
  NUMERIC_ARRAY_VIEW_DECLARE_ASSIGNMENT(/=);
  NUMERIC_ARRAY_VIEW_DECLARE_ASSIGNMENT(%=);
  NUMERIC_ARRAY_VIEW_DECLARE_ASSIGNMENT(&=);
  NUMERIC_ARRAY_VIEW_DECLARE_ASSIGNMENT(|=);
  NUMERIC_ARRAY_VIEW_DECLARE_ASSIGNMENT(^=);
#undef NUMERIC_ARRAY_VIEW_DECLARE_ASSIGNMENT

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE [[nodiscard]] decltype(auto)
  operator()(Idxs... idxs) noexcept;

  template <dim_t M>
  NUMERIC_HOST_DEVICE [[nodiscard]] ArrayView<scalar_t, M>
  broadcast(const Layout<M> &layout) noexcept;

  NUMERIC_HOST_DEVICE [[nodiscard]] ArrayConstView<Scalar, N>
  const_view() const noexcept;

#if NUMERIC_ENABLE_EIGEN
  [[nodiscard]] Eigen::Map<
      Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>, 0,
      Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
  matrix_view() noexcept;
#endif

  using super::memory_type;
  using super::operator();
  using super::raw;
  NUMERIC_HOST_DEVICE [[nodiscard]] scalar_t *raw() noexcept;
  using super::broadcast;
  using super::layout;
  using super::shape;
  using super::size;
  using super::stride;

protected:
  using super::data_;
  using super::layout_;
  using super::memory_type_;

  using super::broadcasted_layout;
  using super::memory_index;

  template <dim_t M, typename Idx, typename... Idxs>
  static NUMERIC_HOST_DEVICE decltype(auto)
  sub_view(ArrayView<scalar_t, M> view, dim_t d, Idx idx,
           Idxs... idxs) noexcept;

  template <dim_t M>
  static NUMERIC_HOST_DEVICE ArrayView<scalar_t, M>
  sub_view(ArrayView<scalar_t, M> view, dim_t) noexcept;
};

template <typename Scalar, dim_t N> struct ArrayTraits<ArrayView<Scalar, N>> {
  static constexpr bool is_array = true;
  static constexpr dim_t dim = N;
  using scalar_t = Scalar;
};

} // namespace numeric::memory

#endif
