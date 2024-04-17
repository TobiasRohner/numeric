#ifndef NUMERIC_MEMORY_CONSTANT_HPP_
#define NUMERIC_MEMORY_CONSTANT_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_base.hpp>
#include <numeric/memory/array_traits.hpp>
#include <numeric/memory/layout.hpp>
#include <numeric/meta/integer_sequence.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::memory {

/**
 * @brief Class representing a constant array.
 *
 * This class represents a constant array with the same value for all elements.
 *
 * @tparam Scalar The type of the elements in the array.
 * @tparam N The dimensionality of the array.
 */
template <typename Scalar, dim_t N>
class Constant : public ArrayBase<Constant<Scalar, N>> {
  using super = ArrayBase<Constant<Scalar, N>>;

public:
  using scalar_t = Scalar;
  static constexpr dim_t dim = N;

  /**
   * @brief Constructs a Constant object.
   *
   * @param shape The shape of the array.
   * @param value The constant value.
   * @param memory_type The memory type of the array.
   */
  NUMERIC_HOST_DEVICE Constant(const Shape<dim> &shape, scalar_t value,
                               MemoryType memory_type = MemoryType::HOST)
      : value_(value), shape_(shape), memory_type_(memory_type) {}

  Constant(const Constant &) = default;
  Constant(Constant &&) = default;
  Constant &operator=(const Constant &) = default;
  Constant &operator=(Constant &&) = default;

  /**
   * @brief Accesses an element or slice of the constant array.
   *
   * @tparam Idxs The indices of the element.
   * @param idxs The indices of the element.
   * @return auto The accessed element.
   */
  template <typename... Idxs>
  NUMERIC_HOST_DEVICE auto operator()(Idxs... idxs) const noexcept {
    if constexpr ((!meta::is_same_v<Idxs, Slice> && ...)) {
      return value_;
    } else {
      return slice(idxs...);
    }
  }

  /**
   * @brief Returns the shape of the constant array.
   */
  NUMERIC_HOST_DEVICE const Shape<N> &shape() const noexcept { return shape_; }

  /**
   * @brief Returns the size of the constant array along the specified
   * dimension.
   *
   * @param idx The index of the dimension.
   * @return dim_t The size of the constant array along the specified dimension.
   */
  NUMERIC_HOST_DEVICE dim_t shape(size_t idx) const noexcept {
    return shape_[idx];
  }

  /**
   * @brief Returns the total number of elements in the constant array.
   */
  NUMERIC_HOST_DEVICE dim_t size() const noexcept { return shape_.size(); }

  /**
   * @brief Returns the memory type of the constant array.
   */
  NUMERIC_HOST_DEVICE MemoryType memory_type() const noexcept {
    return memory_type_;
  }

  /**
   * @brief Broadcasts the constant array to a new shape.
   *
   * @tparam M The dimensionality of the new shape.
   * @param shape The new shape.
   * @return Constant<Scalar, M> The broadcasted array with the new shape.
   */
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

/**
 * @brief Template struct for traits of Constant arrays.
 *
 * This struct provides traits information for Constant arrays.
 *
 * @tparam Scalar The type of the elements in the array.
 * @tparam N The dimensionality of the array.
 */
template <typename Scalar, dim_t N> struct ArrayTraits<Constant<Scalar, N>> {
  static constexpr bool is_array = true;
  using scalar_t = Scalar;
  static constexpr dim_t dim = N;
};

} // namespace numeric::memory

#endif
