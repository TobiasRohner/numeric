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

/**
 * @brief Class for representing a constant view of an array.
 *
 * This class provides a read-only view of an array with specified layout and
 * memory type.
 *
 * @tparam Scalar The data type of the elements in the array.
 * @tparam N The dimensionality of the array.
 */
template <typename Scalar, dim_t N>
class ArrayConstView : public ArrayBase<ArrayConstView<Scalar, N>> {
  using super = ArrayBase<ArrayConstView<Scalar, N>>;

public:
  using scalar_t = Scalar; /**< Data type of the elements in the array. */
  static constexpr dim_t dim = N; /**< Dimensionality of the array. */

  /**
   * @brief Default constructor.
   *
   * Constructs an empty ArrayConstView.
   */
  NUMERIC_HOST_DEVICE ArrayConstView();

  /**
   * @brief Constructs an ArrayConstView with the given data, layout, and memory
   * type.
   *
   * @param data Pointer to the data of the array.
   * @param layout Layout of the array.
   * @param memory_type Memory type of the array.
   */
  NUMERIC_HOST_DEVICE
  ArrayConstView(const scalar_t *data, const Layout<dim> &layout,
                 MemoryType memory_type = MemoryType::UNKNOWN);

  ArrayConstView(const ArrayConstView &) = default;
  ArrayConstView(ArrayConstView &&) = default;
  ArrayConstView &operator=(const ArrayConstView &) = delete;
  ArrayConstView &operator=(ArrayConstView &&) = delete;

  /**
   * @brief Sets the data, layout, and memory type of the ArrayConstView.
   *
   * @param data Pointer to the data of the array.
   * @param layout Layout of the array.
   * @param memory_type Memory type of the array.
   */
  NUMERIC_HOST_DEVICE void
  set(const scalar_t *data, const Layout<dim> &layout,
      MemoryType memory_type = MemoryType::UNKNOWN) noexcept;

  /**
   * @brief Gets the memory type of the array.
   *
   * @return The memory type of the array.
   */
  NUMERIC_HOST_DEVICE MemoryType memory_type() const noexcept;

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
  NUMERIC_HOST_DEVICE decltype(auto) operator()(Idxs... idxs) const noexcept;

  /**
   * @brief Gets a pointer to the raw data of the array.
   *
   * @return Pointer to the raw data of the array.
   */
  NUMERIC_HOST_DEVICE const scalar_t *raw() const noexcept;

  /**
   * @brief Gets the layout of the array.
   *
   * @return Reference to the layout of the array.
   */
  NUMERIC_HOST_DEVICE const Layout<dim> &layout() const noexcept;

  /**
   * @brief Gets the shape of the array.
   *
   * @return The shape of the array.
   */
  NUMERIC_HOST_DEVICE const Shape<N> &shape() const noexcept;

  /**
   * @brief Gets the shape of the array along the specified dimension.
   *
   * @param idx The index of the dimension.
   * @return The size of the array along the specified dimension.
   */
  NUMERIC_HOST_DEVICE dim_t shape(size_t idx) const noexcept;

  /**
   * @brief Gets the stride of the array.
   *
   * @return A reference to the stride of the array.
   */
  NUMERIC_HOST_DEVICE const Stride<N> &stride() const noexcept;

  /**
   * @brief Gets the stride of the array along the specified dimension.
   *
   * @param idx The index of the dimension.
   * @return The stride of the array along the specified dimension.
   */
  NUMERIC_HOST_DEVICE dim_t stride(size_t idx) const noexcept;

  /**
   * @brief Gets the total number of elements in the array.
   *
   * @return The total number of elements in the array.
   */
  NUMERIC_HOST_DEVICE dim_t size() const noexcept;

  /**
   * @brief Broadcasts the array to a new shape.
   *
   * This function creates a new ArrayConstView by broadcasting the current
   * array to the specified shape.
   *
   * @tparam M The dimensionality of the new shape.
   * @param shape The new shape to broadcast to.
   * @return ArrayConstView representing the broadcasted array.
   */
  template <dim_t M>
  NUMERIC_HOST_DEVICE ArrayConstView<scalar_t, M>
  broadcast(const Shape<M> &shape) const noexcept;

#if NUMERIC_ENABLE_EIGEN
  /**
   * @brief Gets an Eigen::Map representing a matrix view of the array.
   *
   * This function returns an Eigen::Map representing a matrix view of the
   * array's data.
   *
   * @return Eigen::Map representing a matrix view of the array's data.
   */
  Eigen::Map<const Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>, 0,
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
  static NUMERIC_HOST_DEVICE ArrayConstView<scalar_t, M>
  sub_view(const ArrayConstView<scalar_t, M> &view, dim_t) noexcept;

  using super::broadcasted_layout;
};

/**
 * @brief Traits struct for ArrayConstView.
 *
 * This struct provides information about ArrayConstView types.
 *
 * @tparam Scalar The data type of the elements in the array.
 * @tparam N The dimensionality of the array.
 */
template <typename Scalar, dim_t N>
struct ArrayTraits<ArrayConstView<Scalar, N>> {
  static constexpr bool is_array =
      true; /**< Indicates whether the type is an array. */
  static constexpr dim_t dim = N; /**< Dimensionality of the array. */
  using scalar_t = Scalar; /**< Data type of the elements in the array. */
};

} // namespace numeric::memory

#endif
