#include <numeric/hip/kernel.hpp>
#include <numeric/hip/safe_call.hpp>
#include <hip/hiprtc.h>


namespace numeric::hip {

Kernel::Kernel(const std::shared_ptr<Module> &module, std::string_view name) : module_(module) {
  NUMERIC_CHECK_HIP(hipModuleGetFunction(&kernel_, module_->module, name.data()));
}

}
