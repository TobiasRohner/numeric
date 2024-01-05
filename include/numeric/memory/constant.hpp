#ifndef NUMERIC_MEMORY_CONSTANT_HPP_
#define NUMERIC_MEMORY_CONSTANT_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_base.hpp>
#include <numeric/memory/array_traits.hpp>
#include <numeric/memory/layout.hpp>
#include <numeric/meta/integer_sequence.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::memory {

template <typename Scalar, dim_t N>
class Constant : ArrayBase<Constant<Scalar, N>> {
  using super = ArrayBase<Constant<Scalar, N>>;

public:
  using scalar_t = Scalar;
  static constexpr dim_t dim = N;

  NUMERIC_HOST_DEVICE Constant(const Layout<dim> &layout, scalar_t value)
      : value_(value), layout_(layout) {}
  Constant(const Constant &) = default;
  Constant(Constant &&) = default;
  Constant &operator=(const Constant &) = default;
  Constant &operator=(Constant &&) = default;

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE auto operator()(Idxs... idxs) const noexcept {
    if constexpr ((!meta::is_same_v<Idxs, Slice> && ...)) {
      return value_;
    } else {
      return slice(idxs...);
    }
  }

private:
  scalar_t value_;
  Layout<dim> layout_;

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE auto slice(Idxs... idxs) const noexcept {
    return slice_impl(meta::index_sequence_for<Idxs...>{}, idxs...);
  }

  template <typename... Idxs, size_t... IdxsIdxs>
  NUMERIC_HOST_DEVICE auto slice_impl(meta::index_sequence<IdxsIdxs...>,
                                      Idxs... idxs) const noexcept {
    static constexpr dim_t new_dim = (0 + ... + meta::is_same_v<Idxs, Slice>);
    Layout<new_dim> new_layout;
    dim_t i = 0;
    const auto get_size = [&](auto sl, size_t size) {
      if constexpr (meta::is_same_v<decltype(sl), Slice>) {
        return sl.size(size);
      } else {
        return 0;
      }
    };
    ((meta::is_same_v<Idxs, Slice> &&
      (new_layout.shape(i++) = get_size(idxs, layout_.shape(IdxsIdxs)))),
     ...);
    return Constant<scalar_t, new_dim>(new_layout, value_);
  }
};

template <typename Scalar, dim_t N> struct ArrayTraits<Constant<Scalar, N>> {
  static constexpr bool is_array = true;
  using scalar_t = Scalar;
  static constexpr dim_t dim = N;
};

} // namespace numeric::memory

#endif
