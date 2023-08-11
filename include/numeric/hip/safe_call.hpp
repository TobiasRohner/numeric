#ifndef NUMERIC_HIP_SAFE_CALL_HPP_
#define NUMERIC_HIP_SAFE_CALL_HPP_

#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

namespace numeric::hip {

void safe_call(const char *file, int line, hipError_t err);
void safe_call(const char *file, int line, hiprtcResult err);

#define NUMERIC_CHECK_HIP(...)                                                 \
  ::numeric::hip::safe_call(__FILE__, __LINE__, __VA_ARGS__)

} // namespace numeric::hip

#endif
