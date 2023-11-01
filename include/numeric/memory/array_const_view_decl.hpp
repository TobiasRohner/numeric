#ifndef NUMERIC_MEMORY_ARRAY_CONST_VIEW_DECL_HPP_
#define NUMERIC_MEMORY_ARRAY_CONST_VIEW_DECL_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_base_decl.hpp>
#include <numeric/memory/layout.hpp>
#include <numeric/memory/memory_type.hpp>
#if NUMERIC_ENABLE_EIGEN
#include <Eigen/Dense>
#endif

namespace numeric::memory {

template <typename Scalar, dim_t N>
class ArrayConstView : public ArrayBase<ArrayConstView<Scalar, N>> {
  using super = ArrayBase<ArrayConstView<Scalar, N>>;

public:
  using scalar_t = Scalar;
  static constexpr dim_t dim = N;

  NUMERIC_HOST_DEVICE ArrayConstView();
  NUMERIC_HOST_DEVICE
  ArrayConstView(const scalar_t *data, const Layout<dim> &layout,
                 MemoryType memory_type = MemoryType::UNKNOWN);
  ArrayConstView(const ArrayConstView &) = default;
  ArrayConstView(ArrayConstView &&) = default;
  ArrayConstView &operator=(const ArrayConstView &) = delete;
  ArrayConstView &operator=(ArrayConstView &&) = delete;

  NUMERIC_HOST_DEVICE void
  set(const scalar_t *data, const Layout<dim> &layout,
      MemoryType memory_type = MemoryType::UNKNOWN) noexcept;

  NUMERIC_HOST_DEVICE [[nodiscard]] MemoryType memory_type() const noexcept;

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE [[nodiscard]] decltype(auto)
  operator()(Idxs... idxs) const noexcept;

  NUMERIC_HOST_DEVICE [[nodiscard]] const scalar_t *raw() const noexcept;
  NUMERIC_HOST_DEVICE [[nodiscard]] const Layout<dim> &layout() const noexcept;
  NUMERIC_HOST_DEVICE [[nodiscard]] const dim_t *shape() const noexcept;
  NUMERIC_HOST_DEVICE [[nodiscard]] dim_t shape(size_t idx) const noexcept;
  NUMERIC_HOST_DEVICE [[nodiscard]] const dim_t *stride() const noexcept;
  NUMERIC_HOST_DEVICE [[nodiscard]] dim_t stride(size_t idx) const noexcept;
  NUMERIC_HOST_DEVICE [[nodiscard]] dim_t size() const noexcept;

  template <dim_t M>
  NUMERIC_HOST_DEVICE [[nodiscard]] ArrayConstView<scalar_t, M>
  broadcast(const Layout<M> &layout) const noexcept;

#if NUMERIC_ENABLE_EIGEN
  [[nodiscard]] Eigen::Map<
      const Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>, 0,
      Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
  matrix_view() const noexcept;
#endif

protected:
  const scalar_t *data_;
  Layout<dim> layout_;
  MemoryType memory_type_;

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE size_t memory_index(Idxs... idxs) const noexcept;

  template <dim_t M, typename Idx, typename... Idxs>
  static NUMERIC_HOST_DEVICE decltype(auto)
  sub_view(const ArrayConstView<scalar_t, M> &view, dim_t d, Idx idx,
           Idxs... idxs) noexcept;

  template <dim_t M>
  static NUMERIC_HOST_DEVICE decltype(auto)
  sub_view(const ArrayConstView<scalar_t, M> &view, dim_t) noexcept;

  using super::broadcasted_layout;
};

template <typename Scalar, dim_t N>
struct ArrayTraits<ArrayConstView<Scalar, N>> {
  static constexpr dim_t dim = N;
  using scalar_t = Scalar;
};

} // namespace numeric::memory

#endif
