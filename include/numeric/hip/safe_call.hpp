#ifndef NUMERIC_HIP_SAFE_CALL_HPP_
#define NUMERIC_HIP_SAFE_CALL_HPP_

#include <numeric/hip/runtime.hpp>

namespace numeric::hip {

void safe_call(const char *file, int line, hipError_t err);
void safe_call(const char *file, int line, hiprtcResult err);

/**
 * @brief Macro for checking the result of a HIP API call.
 *
 * This macro calls the safe_call function to check the result of a HIP API
 * call. If an error occurs, it exits and prints an error with the file name and
 * line number where the error occurred.
 *
 * @param ... HIP API call to check.
 */
#define NUMERIC_CHECK_HIP(...)                                                 \
  ::numeric::hip::safe_call(__FILE__, __LINE__, __VA_ARGS__)

} // namespace numeric::hip

#endif
