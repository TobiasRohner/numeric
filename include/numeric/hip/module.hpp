#ifndef NUMERIC_HIP_MODULE_HPP_
#define NUMERIC_HIP_MODULE_HPP_

#include <numeric/hip/runtime.hpp>
#include <vector>

namespace numeric::hip {

/**
 * @brief Struct representing a HIP module.
 */
struct Module {
  hipModule_t module;       /**< Handle to the HIP module. */
  std::vector<char> binary; /**< Module binary. */

  Module(const std::vector<char> &bin);
  Module(const Module &) = delete;
  Module(Module &&) = default;
  Module &operator=(const Module &) = delete;
  Module &operator=(Module &&) = default;
  ~Module();
};

} // namespace numeric::hip

#endif
