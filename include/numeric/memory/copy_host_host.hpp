#ifndef NUMERIC_MEMORY_COPY_HOST_HOST_HPP_
#define NUMERIC_MEMORY_COPY_HOST_HOST_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_view_decl.hpp>
#include <numeric/memory/copy_kernels.hpp>
#include <numeric/memory/copyer_impl.hpp>

namespace numeric::memory {

/**
 * @brief Class for copying data from host to host.
 *
 * This class provides functionality to copy data from host to host
 * memory.
 *
 * @tparam Scalar The type of the elements in the destination array.
 * @tparam N The dimensionality of the destination array.
 * @tparam Src The type of the source array.
 */
template <typename Scalar, dim_t N, typename Src>
class CopyHostToHost : public CopyerImpl<Scalar, N, Src> {
  using super = CopyerImpl<Scalar, N, Src>;

public:
  /**
   * @brief Performs the copy operation.
   *
   * Copies data from the source array in host memory to the destination
   * array in host memory.
   *
   * @param dst The destination array view.
   * @param src The source array.
   */
  virtual void operator()(ArrayView<Scalar, N> dst,
                          const ArrayBase<Src> &src) override {
    copy_naive_elm(dst, src.derived());
  }
};

} // namespace numeric::memory

#endif
