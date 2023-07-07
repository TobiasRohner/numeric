#ifndef NUMERIC_MEMORY_ARRAY_CONST_VIEW_HPP_
#define NUMERIC_MEMORY_ARRAY_CONST_VIEW_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/layout.hpp>


namespace numeric::memory {

template<typename Scalar, dim_t N>
class ArrayConstView {
public:
  using scalar_t = Scalar;
  static constexpr dim_t dim = N;

  NUMERIC_HOST_DEVICE ArrayConstView(const scalar_t *data, const Layout<dim> &layout) : data_(data), layout_(layout) { }
  NUMERIC_HOST_DEVICE ArrayConstView(const ArrayConstView &) = default;
  NUMERIC_HOST_DEVICE ArrayConstView(ArrayConstView &&) = default;
  NUMERIC_HOST_DEVICE ArrayConstView &operator=(const ArrayConstView &) = default;
  NUMERIC_HOST_DEVICE ArrayConstView &operator=(ArrayConstView &&) = default;

  template<typename... Idxs>
  NUMERIC_HOST_DEVICE const scalar_t &operator()(Idxs... idxs) const noexcept {
    return data_[memory_index(idxs...)];
  }

  NUMERIC_HOST_DEVICE const scalar_t *raw() const noexcept { return data_; }
  NUMERIC_HOST_DEVICE const Layout<dim> &layout() const noexcept { return layout_; }
  NUMERIC_HOST_DEVICE const dim_t *shape() const noexcept { return layout_.shape(); }
  NUMERIC_HOST_DEVICE dim_t shape(size_t idx) const noexcept { return layout_.shape(idx); }
  NUMERIC_HOST_DEVICE const dim_t *stride() const noexcept { return layout_.stride(); }
  NUMERIC_HOST_DEVICE dim_t stride(size_t idx) const noexcept { return layout_.stride(idx); }
  NUMERIC_HOST_DEVICE dim_t size() const noexcept { return layout_.size(); }

protected:
  const scalar_t *data_;
  Layout<dim> layout_;

  template<typename... Idxs>
  NUMERIC_HOST_DEVICE size_t memory_index(Idxs... idxs) const noexcept {
    size_t stride_idx = 0;
    return (... + (idxs*stride(stride_idx++)));
  }
};

}


#endif
