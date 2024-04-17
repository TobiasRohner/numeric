#ifndef NUMERIC_MEMORY_COPYER_IMPL_HPP_
#define NUMERIC_MEMORY_COPYER_IMPL_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_view_decl.hpp>

namespace numeric::memory {

/**
 * @brief Base class for copy operation implementations.
 *
 * This class serves as the base class for various implementations of
 * copy operations between memory locations.
 *
 * @tparam Scalar The type of the elements in the arrays.
 * @tparam N The dimensionality of the arrays.
 * @tparam Src The type of the source array.
 */
template <typename Scalar, dim_t N, typename Src> class CopyerImpl {
public:
  CopyerImpl() = default;
  virtual ~CopyerImpl() = default;

  /**
   * @brief Performs the copy operation.
   *
   * @param dst The destination array view.
   * @param src The source array.
   */
  virtual void operator()(ArrayView<Scalar, N> dst,
                          const ArrayBase<Src> &src) = 0;
};

} // namespace numeric::memory

#endif
