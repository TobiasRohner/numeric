#ifndef NUMERIC_MEMORY_LINSPACE_HPP_
#define NUMERIC_MEMORY_LINSPACE_HPP_

#include <numeric/memory/array_base.hpp>
#include <numeric/memory/array_traits.hpp>
#include <numeric/memory/broadcast.hpp>
#include <numeric/memory/slice.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::memory {

/**
 * @brief Represents a 1-dimensional array of equally spaced values over a
 * specified interval.
 *
 * This class generates an array of equally spaced values over a specified
 * interval. It is similar to NumPy's linspace function.
 *
 * @tparam Scalar The scalar type of the values in the array.
 */
template <typename Scalar> class Linspace : public ArrayBase<Linspace<Scalar>> {
  using super = ArrayBase<Linspace<Scalar>>;

public:
  static constexpr dim_t dim = 1;
  using scalar_t = Scalar;

  /**
   * @brief Constructs a linspace array with the specified parameters.
   *
   * @param start The starting value of the interval.
   * @param stop The ending value of the interval.
   * @param N The number of samples to generate.
   * @param endpoint If true, stop is the last sample. Otherwise, it is not
   * included.
   * @param memory_type The memory type of the array.
   */
  NUMERIC_HOST_DEVICE Linspace(scalar_t start, scalar_t stop, dim_t N,
                               bool endpoint = true,
                               MemoryType memory_type = MemoryType::HOST)
      : start_(start), stop_(stop), N_(N), memory_type_(memory_type),
        endpoint_(endpoint) {}
  Linspace(const Linspace &) = default;
  Linspace &operator=(const Linspace &) = default;

  NUMERIC_HOST_DEVICE dim_t shape(size_t idx) const noexcept { return N_; }
  NUMERIC_HOST_DEVICE MemoryType memory_type() const noexcept {
    return memory_type_;
  }
  NUMERIC_HOST_DEVICE Shape<1> shape() const noexcept { return Shape<1>(N_); }
  NUMERIC_HOST_DEVICE dim_t size() const noexcept { return N_; }

  template <typename Idx>
  NUMERIC_HOST_DEVICE decltype(auto) operator()(Idx idx) const noexcept {
    if constexpr (!meta::is_same_v<Idx, Slice>) {
      return start_ + idx * (stop_ - start_) / (N_ - endpoint_);
    } else {
      if (idx.stop < 0) {
        idx.stop += N_ + 1;
      }
      return Linspace<Scalar>(this->operator()(idx.start),
                              this->operator()(idx.stop - 1),
                              idx.stop - idx.start, endpoint_, memory_type_);
    }
  }

  template <dim_t N>
  NUMERIC_HOST_DEVICE Broadcast<Linspace<Scalar>, N>
  broadcast(const Shape<N> &shape) const noexcept {
    return Broadcast<Linspace<Scalar>, N>(*this, shape);
  }

private:
  scalar_t start_;
  scalar_t stop_;
  dim_t N_;
  MemoryType memory_type_;
  bool endpoint_;
};

/**
 * @brief Generates a linspace array with the specified parameters.
 *
 * @param start The starting value of the interval.
 * @param stop The ending value of the interval.
 * @param N The number of samples to generate.
 * @param endpoint If true, stop is the last sample. Otherwise, it is not
 * included.
 * @param memory_type The memory type of the array.
 * @return Linspace<Scalar> The generated linspace array.
 */
template <typename Scalar>
NUMERIC_HOST_DEVICE Linspace<Scalar>
linspace(Scalar start, Scalar stop, dim_t N, bool endpoint = true,
         MemoryType memory_type = MemoryType::HOST) {
  return Linspace<Scalar>(start, stop, N, endpoint, memory_type);
}

/**
 * @brief Specialization of ArrayTraits for Linspace.
 *
 * Provides information about the Linspace array type.
 */
template <typename Scalar> struct ArrayTraits<Linspace<Scalar>> {
  static constexpr bool is_array = true;
  static constexpr dim_t dim = 1;
  using scalar_t = Scalar;
};

} // namespace numeric::memory

#endif
