#ifndef NUMERIC_MEMORY_COPY_HPP_
#define NUMERIC_MEMORY_COPY_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/copyer.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::memory {

template <typename Scalar, dim_t N, typename Src>
void copy(ArrayView<Scalar, N> dst, const ArrayBase<Src> &src) {
  Copyer<Scalar, N, Src> cpy(dst, src.derived());
  cpy(dst, src);
}

} // namespace numeric::memory

#endif
