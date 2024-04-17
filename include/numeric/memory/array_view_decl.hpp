#ifndef NUMERIC_MEMORY_ARRAY_VIEW_DECL_HPP_
#define NUMERIC_MEMORY_ARRAY_VIEW_DECL_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_const_view_decl.hpp>
#include <numeric/memory/array_traits.hpp>
#if NUMERIC_ENABLE_EIGEN
#include <Eigen/Dense>
#endif

namespace numeric::memory {

/**
 * @brief Class for representing a mutable view of an array.
 *
 * This class provides a mutable view of an array with specified layout and
 * memory type.
 *
 * @tparam Scalar The data type of the elements in the array.
 * @tparam N The dimensionality of the array.
 */
template <typename Scalar, dim_t N>
class ArrayView : public ArrayConstView<Scalar, N> {
  using super = ArrayConstView<Scalar, N>;

public:
  using scalar_t = Scalar; /**< Data type of the elements in the array. */
  static constexpr dim_t dim = N; /**< Dimensionality of the array. */

  /**
   * @brief Default constructor.
   *
   * Constructs an empty ArrayView.
   */
  NUMERIC_HOST_DEVICE ArrayView();

  /**
   * @brief Constructs an ArrayView with the given data, layout, and memory
   * type.
   *
   * @param data Pointer to the data of the array.
   * @param layout Layout of the array.
   * @param memory_type Memory type of the array.
   */
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

  /**
   * @brief Accesses the element at the specified indices.
   *
   * This function allows accessing elements of the array using the provided
   * indices.
   *
   * @tparam Idxs Types of indices.
   * @param idxs Indices to access the element.
   * @return Reference to the accessed element.
   */
  template <typename... Idxs>
  NUMERIC_HOST_DEVICE decltype(auto) operator()(Idxs... idxs) noexcept;

  /**
   * @brief Broadcasts the array to a new shape.
   *
   * This function creates a new ArrayView by broadcasting the current array to
   * the specified shape.
   *
   * @tparam M The dimensionality of the new shape.
   * @param shape The new shape to broadcast to.
   * @return ArrayView representing the broadcasted array.
   */
  template <dim_t M>
  NUMERIC_HOST_DEVICE ArrayView<scalar_t, M>
  broadcast(const Shape<M> &shape) noexcept;

  /**
   * @brief Gets a constant view of the array.
   *
   * @return ArrayConstView representing a constant view of the array.
   */
  NUMERIC_HOST_DEVICE ArrayConstView<Scalar, N> const_view() const noexcept;

#if NUMERIC_ENABLE_EIGEN
  /**
   * @brief Gets an Eigen::Map representing a mutable matrix view of the array.
   *
   * This function returns an Eigen::Map representing a mutable matrix view of
   * the array's data.
   *
   * @return Eigen::Map representing a mutable matrix view of the array's data.
   */
  Eigen::Map<Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>, 0,
             Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
  matrix_view() noexcept;
#endif

  using super::memory_type;
  using super::operator();
  using super::raw;
  /**
   * @brief Gets a pointer to the raw data of the array.
   *
   * This function returns a pointer to the raw data of the array, allowing
   * direct manipulation of the array's elements.
   *
   * @return Pointer to the raw data of the array.
   */
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
  sub_view(ArrayView<scalar_t, M> &view, dim_t d, Idx idx,
           Idxs... idxs) noexcept;

  template <dim_t M>
  static NUMERIC_HOST_DEVICE ArrayView<scalar_t, M>
  sub_view(ArrayView<scalar_t, M> &view, dim_t) noexcept;
};

/**
 * @brief Traits struct for ArrayView.
 *
 * This struct provides information about ArrayView types.
 *
 * @tparam Scalar The data type of the elements in the array.
 * @tparam N The dimensionality of the array.
 */
template <typename Scalar, dim_t N> struct ArrayTraits<ArrayView<Scalar, N>> {
  static constexpr bool is_array =
      true; /**< Indicates whether the type is an array. */
  static constexpr dim_t dim = N; /**< Dimensionality of the array. */
  using scalar_t = Scalar; /**< Data type of the elements in the array. */
};

} // namespace numeric::memory

#endif
