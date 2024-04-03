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
class Constant : public ArrayBase<Constant<Scalar, N>> {
  using super = ArrayBase<Constant<Scalar, N>>;

public:
  using scalar_t = Scalar;
  static constexpr dim_t dim = N;

  NUMERIC_HOST_DEVICE Constant(const Shape<dim> &shape, scalar_t value,
                               MemoryType memory_type = MemoryType::HOST)
      : value_(value), shape_(shape), memory_type_(memory_type) {}
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

  NUMERIC_HOST_DEVICE const Shape<N> &shape() const noexcept { return shape_; }
  NUMERIC_HOST_DEVICE dim_t shape(size_t idx) const noexcept {
    return shape_[idx];
  }
  NUMERIC_HOST_DEVICE dim_t size() const noexcept { return shape_.size(); }
  NUMERIC_HOST_DEVICE MemoryType memory_type() const noexcept {
    return memory_type_;
  }

  template <dim_t M>
  NUMERIC_HOST_DEVICE Constant<Scalar, M>
  broadcast(const Shape<M> &shape) const noexcept {
    return Constant<Scalar, M>(shape, value_, memory_type_);
  }

private:
  scalar_t value_;
  Shape<dim> shape_;
  MemoryType memory_type_;

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE auto slice(Idxs... idxs) const noexcept {
    return slice_impl(meta::index_sequence_for<Idxs...>{}, idxs...);
  }

  template <typename... Idxs, size_t... IdxsIdxs>
  NUMERIC_HOST_DEVICE auto slice_impl(meta::index_sequence<IdxsIdxs...>,
                                      Idxs... idxs) const noexcept {
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
    return Constant<scalar_t, new_dim>(new_shape, value_, memory_type_);
  }
};

template <typename Scalar, dim_t N> struct ArrayTraits<Constant<Scalar, N>> {
  static constexpr bool is_array = true;
  using scalar_t = Scalar;
  static constexpr dim_t dim = N;
};

} // namespace numeric::memory

#endif
