#include <numeric/hip/module.hpp>
#include <numeric/hip/safe_call.hpp>



namespace numeric::hip {

Module::Module(const std::vector<char> &bin) : binary(bin) {
  NUMERIC_CHECK_HIP(hipModuleLoadData(&module, binary.data()));
}

Module::~Module() {
  NUMERIC_CHECK_HIP(hipModuleUnload(module));
}

}
