#ifndef NUMERIC_MEMORY_MESHGRID_HPP_
#define NUMERIC_MEMORY_MESHGRID_HPP_

#include <numeric/memory/array_base.hpp>
#include <numeric/memory/array_traits.hpp>
#include <numeric/memory/broadcast.hpp>
#include <numeric/memory/constant.hpp>
#include <numeric/memory/layout.hpp>
#include <numeric/memory/slice.hpp>
#include <numeric/meta/integer_sequence.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/utils/error.hpp>
#include <numeric/utils/tuple.hpp>

namespace numeric::memory {

/**
 * @brief Represents a meshgrid of a single-dimensional argument.
 *
 * Meshgrid generates coordinate arrays for multidimensional indexing.
 * It is a grid of coordinates corresponding to every point in an N-dimensional
 * space.
 *
 * @tparam Arg Type of the argument.
 * @tparam N Dimensionality of the meshgrid.
 * @tparam Idx Index of the dimension in the meshgrid.
 */
template <typename Arg, dim_t N, dim_t Idx>
class Meshgrid : public ArrayBase<Meshgrid<Arg, N, Idx>> {
  static_assert(ArrayTraits<Arg>::dim == 1, "Meshgrid requires 1d arguments");

  using super = ArrayBase<Meshgrid<Arg, N, Idx>>;

public:
  using scalar_t = typename Arg::scalar_t;
  static constexpr dim_t dim = N;

  NUMERIC_HOST_DEVICE Meshgrid(const Shape<dim> &shape, const Arg &arg)
      : shape_(shape), arg_(arg) {}
  Meshgrid(const Meshgrid &) = default;
  Meshgrid(Meshgrid &&) = default;
  Meshgrid &operator=(const Meshgrid &) = default;
  Meshgrid &operator=(Meshgrid &&) = default;

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE decltype(auto) operator()(Idxs... idxs) const noexcept {
    static constexpr bool use_element_access =
        sizeof...(Idxs) == N && (!meta::is_same_v<Idxs, Slice> && ...);
    if constexpr (use_element_access) {
      return arg_(extract_Idxth(idxs...));
    } else {
      return slice(idxs...);
    }
  }

  NUMERIC_HOST_DEVICE MemoryType memory_type() const noexcept {
    return arg_.memory_type();
  }

  NUMERIC_HOST_DEVICE dim_t shape(size_t idx) const noexcept {
    return shape_[idx];
  }

  NUMERIC_HOST_DEVICE Shape<dim> shape() const noexcept { return shape_; }

  NUMERIC_HOST_DEVICE dim_t size() const noexcept { return shape_.size(); }

  template <dim_t M>
  NUMERIC_HOST_DEVICE Broadcast<Meshgrid<Arg, N, Idx>, M>
  broadcast(const Shape<M> &shape) const noexcept {
    return Broadcast<Meshgrid<Arg, N, Idx>, M>(*this, shape);
  }

private:
  Shape<dim> shape_;
  Arg arg_;

  template <size_t I, typename FirstIdx, typename... Idxs>
  NUMERIC_HOST_DEVICE static auto extract_Idxth_impl(FirstIdx idx,
                                                     Idxs... idxs) noexcept {
    if constexpr (I == Idx) {
      return idx;
    } else {
      return extract_Idxth_impl<I + 1>(idxs...);
    }
  }

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE static auto extract_Idxth(Idxs... idxs) noexcept {
    return extract_Idxth_impl<0>(idxs...);
  }

  template <typename... Idxs, size_t... IdxsIdxs>
  NUMERIC_HOST_DEVICE decltype(auto)
  slice_impl(meta::index_sequence<IdxsIdxs...>, Idxs... idxs) const noexcept {
    static constexpr dim_t num_slices_in_idxs =
        (0 + ... + meta::is_same_v<Idxs, Slice>);
    static constexpr dim_t new_dim = N - sizeof...(Idxs) + num_slices_in_idxs;
    const auto get_size = [&](auto sl, size_t size) {
      if constexpr (meta::is_same_v<decltype(sl), Slice>) {
        return sl.size(size);
      } else {
        return 0;
      }
    };
    Shape<new_dim> new_shape;
    dim_t i = 0;
    ((meta::is_same_v<Idxs, Slice> &&
      (new_shape[i++] = get_size(idxs, shape_[IdxsIdxs]))),
     ...);
    for (; i < new_dim; ++i) {
      new_shape[i] = shape_[sizeof...(Idxs) + i];
    }
    static constexpr bool is_constant =
        ((IdxsIdxs == Idx && !meta::is_same_v<Idxs, Slice>) || ...);
    if constexpr (is_constant) {
      using scalar_t = typename ArrayTraits<Arg>::scalar_t;
      const scalar_t value = arg_(extract_Idxth(idxs...));
      return Constant<scalar_t, new_dim>(new_shape, value, arg_.memory_type());
    } else {
      static constexpr dim_t ndim_left =
          (0 + ... + (meta::is_same_v<Idxs, Slice> && IdxsIdxs < Idx));
      if constexpr (sizeof...(Idxs) > Idx) {
        auto arg_slice = arg_(extract_Idxth(idxs...));
        return Meshgrid<decltype(arg_slice), new_dim, ndim_left>(new_shape,
                                                                 arg_slice);
      } else {
        return Meshgrid<Arg, new_dim, ndim_left>(new_shape, arg_);
      }
    }
  }

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE decltype(auto) slice(Idxs... idxs) const noexcept {
    return slice_impl(meta::index_sequence_for<Idxs...>{}, idxs...);
  }
};

/**
 * @brief Traits for the Meshgrid class.
 *
 * Specialization of ArrayTraits for Meshgrid class.
 *
 * @tparam Arg Type of the argument.
 * @tparam N Dimensionality of the meshgrid.
 * @tparam Idx Index of the dimension in the meshgrid.
 */
template <typename Arg, dim_t N, dim_t Idx>
struct ArrayTraits<Meshgrid<Arg, N, Idx>> {
  static constexpr bool is_array = true;
  static constexpr dim_t dim = N;
  using scalar_t = typename ArrayTraits<Arg>::scalar_t;
};

namespace detail {

template <size_t... Idxs, typename... Args>
NUMERIC_HOST_DEVICE auto meshgrid_impl(meta::index_sequence<Idxs...>,
                                       const Args &...args) {
  static constexpr size_t N = sizeof...(Args);
  const Shape<N> shape(args.shape(Idxs)...);
  return utils::Tuple<Meshgrid<Args, N, Idxs>...>(
      Meshgrid<Args, N, Idxs>(shape, args)...);
}

} // namespace detail

/**
 * @brief Generates meshgrids for given arguments.
 *
 * This function generates meshgrids for each argument provided.
 *
 * @tparam Args Types of arguments.
 * @param args Arguments for generating meshgrids.
 * @return Tuple containing meshgrids for each argument.
 */
template <typename... Args>
NUMERIC_HOST_DEVICE auto meshgrid(const ArrayBase<Args> &...args) {
  return detail::meshgrid_impl(meta::make_index_sequence<sizeof...(Args)>(),
                               args.derived()...);
}

} // namespace numeric::memory

#endif
