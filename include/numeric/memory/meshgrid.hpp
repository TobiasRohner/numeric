#ifndef NUMERIC_MEMORY_MESHGRID_HPP_
#define NUMERIC_MEMORY_MESHGRID_HPP_

#include <numeric/memory/array_base.hpp>
#include <numeric/memory/array_traits.hpp>
#include <numeric/memory/constant.hpp>
#include <numeric/memory/layout.hpp>
#include <numeric/memory/slice.hpp>
#include <numeric/meta/integer_sequence.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/utils/tuple.hpp>

namespace numeric::memory {

template <typename Arg, dim_t N, dim_t Idx>
class Meshgrid : public ArrayBase<Meshgrid<Arg, N, Idx>> {
  static_assert(ArrayTraits<Arg>::dim == 1, "Meshgrid requires 1d arguments");

  using super = ArrayBase<Meshgrid<Arg, N, Idx>>;

public:
  static constexpr dim_t dim = N;

  Meshgrid(const Arg &arg) : layout_(), arg_(arg) {
    for (dim_t i = 0; i < dim; ++i) {
      layout_.shape(i) = 1;
    }
    layout_.shape(Idx) = arg_.shape(0);
  }

  template <typename... Idxs> auto operator()(Idxs... idxs) const noexcept {
    if constexpr (use_direct_access(idxs...)) {
      return arg_(extract_Idxth(idxs...));
    } else {
      return slice(idxs...);
    }
  }

private:
  Layout<dim> layout_;
  Arg arg_;

  template <typename... Idxs, size_t... IdxsIdxs>
  static constexpr bool
  use_direct_access_impl(meta::index_sequence<IdxsIdxs...>,
                         Idxs... idxs) noexcept {
    return ((!meta::is_same_v<Idxs, Slice> || IdxsIdxs == Idx) && ...);
  }

  template <typename... Idxs>
  static constexpr bool use_direct_access(Idxs... idxs) noexcept {
    return use_direct_access_impl(meta::index_sequence_for<Idxs...>{}, idxs...);
  }

  template <size_t I, typename FirstIdx, typename... Idxs>
  static auto extract_Idxth_impl(FirstIdx idx, Idxs... idxs) noexcept {
    static_assert(I <= sizeof...(Idxs),
                  "Trying to access out of bounds element of parameter pack");
    if constexpr (I == Idx) {
      return idx;
    } else {
      return extract_Idxth_impl<I + 1>(idxs...);
    }
  }

  template <typename... Idxs> static auto extract_Idxth(Idxs... idxs) noexcept {
    return extract_Idxth_impl<0>(idxs...);
  }

  template <typename... Idxs, size_t... IdxsIdxs>
  auto slice_impl(meta::index_sequence<IdxsIdxs...>,
                  Idxs... idxs) const noexcept {
    static constexpr dim_t ndim_left =
        (0 + ... + (meta::is_same_v<Idxs, Slice> && IdxsIdxs < Idx));
    static constexpr dim_t ndim_right =
        (0 + ... + (meta::is_same_v<Idxs, Slice> && IdxsIdxs > Idx));
    // TODO: Implement
  }

  template <typename... Idxs> auto slice(Idxs... idxs) const noexcept {
    return slice_impl(meta::index_sequence_for<Idxs...>{}, idxs...);
  }
};

template <typename Arg, dim_t N, dim_t Idx>
struct ArrayTraits<Meshgrid<Arg, N, Idx>> {
  static constexpr bool is_array = true;
  static constexpr dim_t dim = N;
  using scalar_t = typename ArrayTraits<Arg>::scalar_t;
};

namespace detail {

template <size_t... Idxs, typename... Args>
auto meshgrid_impl(meta::index_sequence<Idxs...>, const Args &...args) {
  static constexpr size_t N = sizeof...(Args);
  return utils::Tuple<Meshgrid<Args, N, Idxs>...>(
      Meshgrid<Args, N, Idxs>(args)...);
}

} // namespace detail

template <typename... Args> auto meshgrid(const ArrayBase<Args> &...args) {
  return detail::meshgrid_impl(meta::make_index_sequence<sizeof...(Args)>(),
                               args.derived()...);
}

} // namespace numeric::memory

#endif
