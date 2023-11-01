#ifndef NUMERIC_MEMORY_COPYER_IMPL_HPP_
#define NUMERIC_MEMORY_COPYER_IMPL_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_view_decl.hpp>

namespace numeric::memory {

template <typename Scalar, dim_t N, typename Src> class CopyerImpl {
public:
  CopyerImpl() = default;
  virtual ~CopyerImpl() = default;

  virtual void operator()(ArrayView<Scalar, N> dst,
                          const ArrayBase<Src> &src) = 0;
};

} // namespace numeric::memory

#endif
