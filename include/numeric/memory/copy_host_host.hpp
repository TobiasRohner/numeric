#ifndef NUMERIC_MEMORY_COPY_HOST_HOST_HPP_
#define NUMERIC_MEMORY_COPY_HOST_HOST_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_view_decl.hpp>
#include <numeric/memory/copy_kernels.hpp>

namespace numeric::memory {

template <typename Scalar, dim_t N, typename Src> class CopyHostToHost {
public:
  void operator()(ArrayView<Scalar, N> dst, const ArrayBase<Src> &src) const {
    copy_naive_elm(dst, src.derived());
  }
};

} // namespace numeric::memory

#endif
