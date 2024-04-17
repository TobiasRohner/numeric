#ifndef NUMERIC_MEMORY_BROADCAST_DECL_HPP_
#define NUMERIC_MEMORY_BROADCAST_DECL_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_base_decl.hpp>
#include <numeric/memory/array_traits.hpp>
#include <numeric/memory/slice.hpp>
#include <numeric/meta/integer_sequence.hpp>
#include <numeric/utils/tuple.hpp>

namespace numeric::memory {

/**
 * @brief Class for broadcasting an array to a specified shape.
 *
 * This class broadcasts an array to a specified shape, replicating elements
 * along dimensions as needed.
 *
 * @tparam Src The type of the source array.
 * @tparam N The dimensionality of the broadcasted array.
 */
template <typename Src, dim_t N> class Broadcast {
public:
  static constexpr dim_t dim = N;
  using scalar_t = typename ArrayTraits<Src>::scalar_t;

  /**
   * @brief Constructs a Broadcast object.
   *
   * @param src The source array.
   * @param shape The shape to which the source array is broadcasted.
   */
  NUMERIC_HOST_DEVICE Broadcast(const ArrayBase<Src> &src,
                                const Shape<N> &shape);

  Broadcast(const Broadcast &) = default;
  Broadcast(Broadcast &&) = default;
  Broadcast &operator=(const Broadcast &) = default;
  Broadcast &operator=(Broadcast &&) = default;

  /**
   * @brief Returns the memory type of the broadcasted array.
   */
  NUMERIC_HOST_DEVICE MemoryType memory_type() const noexcept;

  /**
   * @brief Accesses an element or slice of the broadcasted array.
   *
   * @tparam Idxs The indices of the element.
   * @param idxs The indices of the element.
   * @return decltype(auto) The accessed element.
   */
  template <typename... Idxs>
  NUMERIC_HOST_DEVICE decltype(auto) operator()(Idxs... idxs) const noexcept;

  /**
   * @brief Returns the shape of the broadcasted array.
   */
  NUMERIC_HOST_DEVICE const Shape<N> &shape() const noexcept;

  /**
   * @brief Returns the size of the broadcasted array along the specified
   * dimension.
   *
   * @param idx The index of the dimension.
   * @return dim_t The size of the broadcasted array along the specified
   * dimension.
   */
  NUMERIC_HOST_DEVICE dim_t shape(size_t idx) const noexcept;

  /**
   * @brief Returns the size of the broadcasted array.
   */
  NUMERIC_HOST_DEVICE dim_t size() const noexcept;

  /**
   * @brief Broadcasts the array to a new shape.
   *
   * @tparam M The dimensionality of the new shape.
   * @param shape The new shape.
   * @return Broadcast<Src, M> The broadcasted array with the new shape.
   */
  template <dim_t M>
  NUMERIC_HOST_DEVICE Broadcast<Src, M>
  broadcast(const Shape<M> &shape) const noexcept;

private:
  Src src_;
  Shape<N> shape_;

  template <typename Idx, typename... Idxs>
  NUMERIC_HOST_DEVICE scalar_t element(Idx idx, Idxs... idxs) const noexcept;

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE decltype(auto) sub_view(Idxs... idxs) const noexcept;
  template <typename... Idxs, size_t... IdxsIdxs>
  NUMERIC_HOST_DEVICE decltype(auto)
  sub_view_impl(meta::index_sequence<IdxsIdxs...>, Idxs... idxs) const noexcept;
  template <typename... Idxs, size_t... IdxsIdxs, size_t... IdxsIdxsBroadcast,
            size_t... IdxsIdxsSrc>
  NUMERIC_HOST_DEVICE decltype(auto)
  sub_view_impl(const utils::Tuple<Idxs...> &idxs,
                meta::index_sequence<IdxsIdxs...>,
                meta::index_sequence<IdxsIdxsBroadcast...>,
                meta::index_sequence<IdxsIdxsSrc...>) const noexcept;
};

/**
 * @brief Template struct for traits of Broadcast arrays.
 *
 * This struct provides traits information for Broadcast arrays.
 *
 * @tparam Src The type of the source array.
 * @tparam N The dimensionality of the broadcasted array.
 */
template <typename Src, dim_t N> struct ArrayTraits<Broadcast<Src, N>> {
  static constexpr bool is_array = true;
  static constexpr dim_t dim = N;
  using scalar_t = typename ArrayTraits<Src>::scalar_t;
};

/**
 * @brief Function to broadcast an array to a specified shape.
 *
 * This function broadcasts an array to a specified shape.
 *
 * @tparam Src The type of the source array.
 * @tparam N The dimensionality of the broadcasted array.
 * @param src The source array.
 * @param shape The shape to which the source array is broadcasted.
 * @return Broadcast<Src, N> The broadcasted array.
 */
template <typename Src, dim_t N>
NUMERIC_HOST_DEVICE Broadcast<Src, N> broadcast(const ArrayBase<Src> &src,
                                                const Shape<N> &shape) noexcept;

} // namespace numeric::memory

#endif
