#ifndef NUMERIC_MEMORY_ARRAY_VIEW_HPP_
#define NUMERIC_MEMORY_ARRAY_VIEW_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_const_view.hpp>

namespace numeric::memory {

template <typename Scalar, dim_t N>
class ArrayView : public ArrayConstView<Scalar, N> {
  using super = ArrayConstView<Scalar, N>;

public:
  using scalar_t = Scalar;
  static constexpr dim_t dim = N;

  NUMERIC_HOST_DEVICE ArrayView() : super(nullptr, {}, MemoryType::UNKNOWN) {}
  NUMERIC_HOST_DEVICE ArrayView(scalar_t *data, const Layout<dim> &layout,
                                MemoryType memory_type = MemoryType::UNKNOWN)
      : super(data, layout, memory_type) {}
  NUMERIC_HOST_DEVICE ArrayView(const ArrayView &) = default;
  NUMERIC_HOST_DEVICE ArrayView(ArrayView &&) = default;
  NUMERIC_HOST_DEVICE ArrayView &operator=(const ArrayView &) = default;
  NUMERIC_HOST_DEVICE ArrayView &operator=(ArrayView &&) = default;

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE [[nodiscard]] scalar_t &
  operator()(Idxs... idxs) noexcept {
    return raw()[memory_index(idxs...)];
  }

  template <dim_t M>
  NUMERIC_HOST_DEVICE [[nodiscard]] ArrayView<scalar_t, M>
  broadcast(const Layout<M> &layout) noexcept {
    const Layout<M> new_layout = broadcasted_shape(layout);
    return ArrayView<scalar_t, M>(raw(), new_layout, memory_type());
  }

  NUMERIC_HOST_DEVICE [[nodiscard]] ArrayConstView<Scalar, N>
  const_view() const noexcept {
    return *this;
  }

  using super::memory_type;
  using super::operator();
  using super::raw;
  NUMERIC_HOST_DEVICE [[nodiscard]] scalar_t *raw() noexcept {
    return const_cast<scalar_t *>(data_);
  }
  using super::broadcast;
  using super::layout;
  using super::shape;
  using super::size;
  using super::stride;

protected:
  using super::data_;
  using super::layout_;
  using super::memory_type_;

  using super::broadcasted_shape;
  using super::memory_index;
};

} // namespace numeric::memory

#endif
