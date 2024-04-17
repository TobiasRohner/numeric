#ifndef NUMERIC_MEMORY_ARRAY_HPP_
#define NUMERIC_MEMORY_ARRAY_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/allocator.hpp>
#include <numeric/memory/array_traits.hpp>
#include <numeric/memory/array_view.hpp>

namespace numeric::memory {

/**
 * @brief Class for representing a dynamically allocated array.
 *
 * This class provides functionality for managing dynamically allocated arrays
 * with specified layout and memory type.
 *
 * @tparam Scalar The data type of the elements in the array.
 * @tparam N The dimensionality of the array.
 */
template <typename Scalar, dim_t N> class Array : public ArrayView<Scalar, N> {
  using super = ArrayView<Scalar, N>;

public:
  using scalar_t = Scalar; /**< Data type of the elements in the array. */
  static constexpr dim_t dim = N; /**< Dimensionality of the array. */

  /**
   * @brief Default constructor.
   *
   * Constructs an empty Array with no memory allocation.
   */
  Array() : super(nullptr, {}, MemoryType::UNKNOWN), alloc_(nullptr) {}

  /**
   * @brief Constructs an Array with the given shape and allocator.
   *
   * @param shape The shape of the array.
   * @param alloc The allocator used for memory allocation.
   */
  Array(const Shape<dim> &shape, Allocator<scalar_t> alloc)
      : super(alloc.allocate(shape.size()), Layout<dim>(shape),
              alloc.memory_type()),
        alloc_(alloc) {}

  /**
   * @brief Constructs an Array with the given shape and memory type.
   *
   * @param shape The shape of the array.
   * @param mem_type The memory type of the array.
   */
  explicit Array(const Shape<dim> &shape,
                 MemoryType mem_type = MemoryType::HOST)
      : Array(shape, Allocator<scalar_t>(mem_type)) {}

  Array(const Array &other) : Array(other.shape(), other.memory_type_) {
    *this = other;
  }
  template <typename Src>
  Array(const ArrayBase<Src> &src) : Array(src.shape(), src.memory_type()) {
    *this = src;
  }

  Array(Array &&other)
      : super(other.raw(), other.layout_, other.memory_type_),
        alloc_(std::move(other.alloc_)) {
    other.data_ = nullptr;
  }
  Array &operator=(Array &&other) {
    if (raw()) {
      alloc_.deallocate(raw(), size());
    }
    alloc_ = other.alloc_;
    data_ = other.data_;
    layout_ = other.layout_;
    memory_type_ = other.memory_type_;
    other.data_ = nullptr;
    return *this;
  }
  Array &operator=(const Array &other) {
    super::operator=(other);
    return *this;
  }
  template <typename Src> Array &operator=(const ArrayBase<Src> &src) {
    super::operator=(src);
    return *this;
  }
  Array &operator=(Scalar val) {
    super::operator=(val);
    return *this;
  }
#define NUMERIC_ARRAY_DEFINE_ASSIGNMENT(op)                                    \
  template <typename Src> Array &operator op(const ArrayBase<Src> &src) {      \
    super::operator op(src);                                                   \
    return *this;                                                              \
  }                                                                            \
  Array &operator op(Scalar val) {                                             \
    super::operator op(val);                                                   \
    return *this;                                                              \
  }
  NUMERIC_ARRAY_DEFINE_ASSIGNMENT(+=);
  NUMERIC_ARRAY_DEFINE_ASSIGNMENT(-=);
  NUMERIC_ARRAY_DEFINE_ASSIGNMENT(*=);
  NUMERIC_ARRAY_DEFINE_ASSIGNMENT(/=);
  NUMERIC_ARRAY_DEFINE_ASSIGNMENT(%=);
  NUMERIC_ARRAY_DEFINE_ASSIGNMENT(&=);
  NUMERIC_ARRAY_DEFINE_ASSIGNMENT(|=);
  NUMERIC_ARRAY_DEFINE_ASSIGNMENT(^=);
#undef NUMERIC_ARRAY_DEFINE_ASSIGNMENT

  ~Array() {
    if (raw()) {
      alloc_.deallocate(raw(), size());
    }
  }

  /**
   * @brief Gets a mutable view of the array.
   *
   * @return ArrayView representing a mutable view of the array.
   */
  ArrayView<Scalar, N> view() noexcept { return *this; }

  using super::const_view;
  using super::memory_type;
  using super::operator();
  using super::broadcast;
  using super::layout;
  using super::raw;
  using super::shape;
  using super::size;
  using super::stride;

protected:
  Allocator<scalar_t> alloc_;
  using super::data_;
  using super::layout_;
  using super::memory_type_;

  using super::memory_index;
};

/**
 * @brief Traits struct for Array.
 *
 * This struct provides information about Array types.
 *
 * @tparam Scalar The data type of the elements in the array.
 * @tparam N The dimensionality of the array.
 */
template <typename Scalar, dim_t N> struct ArrayTraits<Array<Scalar, N>> {
  static constexpr bool is_array =
      true; /**< Indicates whether the type is an array. */
  static constexpr dim_t dim = N; /**< Dimensionality of the array. */
  using scalar_t = Scalar; /**< Data type of the elements in the array. */
};

} // namespace numeric::memory

#endif
