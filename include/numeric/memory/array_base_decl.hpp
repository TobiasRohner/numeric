#ifndef NUMERIC_MEMORY_ARRAY_BASE_DECL_HPP_
#define NUMERIC_MEMORY_ARRAY_BASE_DECL_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_traits.hpp>
#include <numeric/memory/layout.hpp>
#include <numeric/memory/memory_type.hpp>

namespace numeric::memory {

/**
 * @brief Base class for arrays with common memory management operations.
 *
 * This class provides common operations for arrays, such as obtaining shape,
 * memory type, and size.
 *
 * @tparam Derived The derived array class.
 */
template <typename Derived> class ArrayBase {
public:
  static constexpr dim_t dim =
      ArrayTraits<Derived>::dim; /**< Dimensionality of the array. */

  /**
   * @brief Gets the shape of the array along the specified dimension.
   *
   * @param idx The index of the dimension.
   * @return The size of the array along the specified dimension.
   */
  NUMERIC_HOST_DEVICE dim_t shape(size_t idx) const noexcept;

  /**
   * @brief Gets a reference to the derived array object.
   *
   * @return A reference to the derived array object.
   */
  NUMERIC_HOST_DEVICE Derived &derived() noexcept;

  /**
   * @brief Gets a const reference to the derived array object.
   *
   * @return A const reference to the derived array object.
   */
  NUMERIC_HOST_DEVICE const Derived &derived() const noexcept;

  /**
   * @brief Gets the memory type of the array.
   *
   * @return The memory type of the array.
   */
  NUMERIC_HOST_DEVICE MemoryType memory_type() const noexcept;

  /**
   * @brief Gets the shape of the array.
   *
   * @return The shape of the array.
   */
  NUMERIC_HOST_DEVICE Shape<dim> shape() const noexcept;

  /**
   * @brief Gets the total number of elements in the array.
   *
   * @return The total number of elements in the array.
   */
  NUMERIC_HOST_DEVICE dim_t size() const noexcept;

protected:
  /**
   * @brief Generates a broadcasted layout from one layout to another.
   *
   * @tparam N The dimensionality of the original layout.
   * @tparam M The dimensionality of the target layout.
   * @param from The original layout.
   * @param to The target shape.
   * @return The broadcasted layout.
   */
  template <dim_t N, dim_t M>
  static NUMERIC_HOST_DEVICE Layout<M>
  broadcasted_layout(const Layout<N> &from, const Shape<M> &to) noexcept;
};

} // namespace numeric::memory

#endif
