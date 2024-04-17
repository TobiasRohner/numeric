#ifndef NUMERIC_MEMORY_COPY_HPP_
#define NUMERIC_MEMORY_COPY_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/copyer.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::memory {

/**
 * @brief Copy data from source to destination.
 *
 * This function provides a convenient way to copy data from a source array
 * to a destination array optionally residing in different memory locations.
 *
 * @tparam Scalar The type of the elements in the destination array.
 * @tparam N The dimensionality of the destination array.
 * @tparam Src The type of the source array.
 *
 * @param dst The destination array view.
 * @param src The source array.
 */
template <typename Scalar, dim_t N, typename Src>
void copy(ArrayView<Scalar, N> dst, const ArrayBase<Src> &src) {
  Copyer<Scalar, N, Src> cpy(dst, src.derived());
  cpy(dst, src);
}

} // namespace numeric::memory

#endif
