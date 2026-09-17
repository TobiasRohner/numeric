#ifndef NUMERIC_MEMORY_ARRAY_HPP_
#define NUMERIC_MEMORY_ARRAY_HPP_

#include <limits>
#include <numeric/config.hpp>
#include <numeric/memory/allocator.hpp>
#include <numeric/memory/array_traits.hpp>
#include <numeric/memory/array_view.hpp>
#include <numeric/utils/error.hpp>
#include <random>
#include <type_traits>

namespace numeric::memory {

namespace detail {

template <dim_t N, dim_t M>
NUMERIC_HOST_DEVICE Shape<N> assign_shape_impl(const Shape<M> &src_shape) {
  if constexpr (N == M) {
    return src_shape;
  } else if constexpr (N < M) {
    Shape<N> result;
    for (dim_t i = 0; i < M - N; ++i) {
      NUMERIC_ERROR_IF(src_shape[i] != 1,
                       "Cannot assign to Array with fewer dimensions: leading "
                       "dimensions of source must be 1");
    }
    for (dim_t i = 0; i < N; ++i) {
      result[i] = src_shape[M - N + i];
    }
    return result;
  } else {
    Shape<N> result;
    for (dim_t i = 0; i < N - M; ++i) {
      result[i] = 1;
    }
    for (dim_t i = 0; i < M; ++i) {
      result[N - M + i] = src_shape[i];
    }
    return result;
  }
}

} // namespace detail

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

  /*
   * @brief Generates an Array with uniformly distributed elements
   *
   * @param min Minimum value of the distribution
   * @param max Maximum value (inclusive) of the distribution
   * @param shape Shape of the generated Array
   * @param rng Any random number generator
   */
  template <typename RNG>
  static Array<Scalar, N> uniform(scalar_t min, scalar_t max,
                                  const Shape<dim> &shape, RNG &rng) {
    static_assert(
        std::numeric_limits<scalar_t>::is_specialized,
        "std::numeric_limits is not specialized for given scalar type");
    using dist_t = std::conditional_t<std::numeric_limits<scalar_t>::is_integer,
                                      std::uniform_int_distribution<scalar_t>,
                                      std::uniform_real_distribution<scalar_t>>;
    Array<Scalar, N> a(shape, MemoryType::HOST);
    dist_t dist(min, max);
    for (dim_t i = 0; i < shape.size(); ++i) {
      a.raw()[i] = dist(rng);
    }
    return a;
  }

  /*
   * @brief Generates an Array with normal distributed elements
   *
   * @param mu Mean of the distribution
   * @param sigma Standard deviation of the distribution
   * @param shape Shape of the generated Array
   * @param rng Any random number generator
   */
  template <typename RNG>
  static Array<Scalar, N> normal(scalar_t mu, scalar_t sigma,
                                 const Shape<dim> &shape, RNG &rng) {
    Array<Scalar, N> a(shape, MemoryType::HOST);
    std::normal_distribution<scalar_t> dist(mu, sigma);
    for (dim_t i = 0; i < shape.size(); ++i) {
      a.raw()[i] = dist(rng);
    }
    return a;
  }

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
    if (!raw()) {
      *this = Array(other.shape(), other.memory_type());
    }
    super::operator=(other);
    return *this;
  }
  template <typename Src> Array &operator=(const ArrayBase<Src> &src) {
    if (!raw()) {
      using SrcTraits = ArrayTraits<Src>;
      constexpr dim_t SrcDim = SrcTraits::dim;
      const Shape<N> new_shape =
          detail::assign_shape_impl<N, SrcDim>(src.shape());
      *this = Array(new_shape, src.memory_type());
    }
    super::operator=(src);
    return *this;
  }
  Array &operator=(Scalar val) {
    if (!raw()) {
      Shape<dim> shape;
      for (dim_t i = 0; i < N; ++i) {
        shape[i] = 1;
      }
      *this = Array(shape, MemoryType::HOST);
    }
    super::operator=(val);
    return *this;
  }
#define NUMERIC_ARRAY_DEFINE_ASSIGNMENT(op)                                    \
  template <typename Src> Array &operator op(const ArrayBase<Src> &src) {      \
    NUMERIC_ERROR_IF(!raw(),                                                   \
                     "Cannot use compound assignment on uninitialized Array"); \
    super::operator op(src);                                                   \
    return *this;                                                              \
  }                                                                            \
  Array &operator op(Scalar val) {                                             \
    NUMERIC_ERROR_IF(!raw(),                                                   \
                     "Cannot use compound assignment on uninitialized Array"); \
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
   * @brief Move this array to the given memory location
   *
   * If the memory is already located in the requested location, this does
   * nothing. Otherwise it allocates new memory and copies the data over.
   */
  void to(MemoryType mem_type) {
    if (memory_type() == mem_type) {
      return;
    }
    Array<Scalar, N> other(shape(), mem_type);
    other = *this;
    *this = std::move(other);
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
