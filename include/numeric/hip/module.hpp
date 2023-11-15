#ifndef NUMERIC_HIP_MODULE_HPP_
#define NUMERIC_HIP_MODULE_HPP_

#include <numeric/hip/runtime.hpp>
#include <vector>

namespace numeric::hip {

struct Module {
  hipModule_t module;
  std::vector<char> binary;

  Module(const std::vector<char> &bin);
  Module(const Module &) = delete;
  Module(Module &&) = default;
  Module &operator=(const Module &) = delete;
  Module &operator=(Module &&) = default;
  ~Module();
};

} // namespace numeric::hip

#endif
