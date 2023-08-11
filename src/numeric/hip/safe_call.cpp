#include <iostream>
#include <numeric/hip/safe_call.hpp>

namespace numeric::hip {

void safe_call(const char *file, int line, hipError_t err) {
  if (err != hipSuccess) {
    std::cerr << file << ":" << line << ": " << hipGetErrorString(err)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

void safe_call(const char *file, int line, hiprtcResult err) {
  if (err != HIPRTC_SUCCESS) {
    std::cerr << file << ":" << line << ": " << hiprtcGetErrorString(err)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

} // namespace numeric::hip
