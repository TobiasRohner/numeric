#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric/config.hpp>
#include <numeric/hip/program.hpp>
#include <numeric/hip/safe_call.hpp>

namespace numeric::hip {

std::vector<std::string_view> Program::numeric_headers_ = {
    "numeric/config.hpp",
    "numeric/hip/runtime.hpp",
    "numeric/math/constants.hpp",
    "numeric/math/functions.hpp",
    "numeric/math/polynomial.hpp",
    "numeric/math/dihedral_group.hpp",
    "numeric/memory/array_traits.hpp",
    "numeric/memory/array_base_decl.hpp",
    "numeric/memory/array_base.hpp",
    "numeric/memory/array_const_view_decl.hpp",
    "numeric/memory/array_const_view.hpp",
    "numeric/memory/array_view_decl.hpp",
    "numeric/memory/array_view.hpp",
    "numeric/memory/constant.hpp",
    "numeric/memory/linspace.hpp",
    "numeric/memory/meshgrid.hpp",
    "numeric/memory/broadcast.hpp",
    "numeric/memory/broadcast_decl.hpp",
    "numeric/memory/layout.hpp",
    "numeric/memory/shape.hpp",
    "numeric/memory/stride.hpp",
    "numeric/memory/slice.hpp",
    "numeric/memory/memory_type.hpp",
    "numeric/memory/copy_kernels.hpp",
    "numeric/memory/array_op.hpp",
    "numeric/utils/error.hpp",
    "numeric/utils/tuple.hpp",
    "numeric/utils/forward.hpp",
    "numeric/meta/meta.hpp",
    "numeric/meta/type_tag.hpp",
    "numeric/meta/integer_sequence.hpp",
    "numeric/mesh/ref_el_point.hpp",
    "numeric/mesh/ref_el_segment.hpp",
    "numeric/mesh/ref_el_tria.hpp",
    "numeric/mesh/ref_el_quad.hpp",
    "numeric/mesh/ref_el_tetra.hpp",
    "numeric/mesh/ref_el_cube.hpp",
    "numeric/mesh/element_traits.hpp",
    "numeric/mesh/element_base.hpp",
    "numeric/mesh/point.hpp",
    "numeric/mesh/segment.hpp",
    "numeric/mesh/tria.hpp",
    "numeric/mesh/quad.hpp",
    "numeric/mesh/tetra.hpp",
    "numeric/mesh/cube.hpp",
    "numeric/mesh/elements.hpp",
    "numeric/math/basis_lagrange.hpp",
    "numeric/math/basis_lagrange_segment.hpp",
    "numeric/math/basis_lagrange_tria.hpp",
    "numeric/math/basis_lagrange_quad.hpp",
    "numeric/math/basis_lagrange_tetra.hpp",
    "numeric/math/basis_lagrange_cube.hpp",
    "numeric/math/basis_lagrange_specialization.hpp",
    "numeric/math/fes/basis_base.hpp",
    "numeric/math/fes/basis_h1.hpp",
    "numeric/math/fes/basis_l2.hpp",
    "numeric/equations/fem/load_element_vector.hpp",
    "numeric/equations/fem/diffusion_element_matrix.hpp"};

Program::Header::Header(std::string_view name_, std::string_view src_)
    : name(name_), src(src_) {}

Program::Program(std::string src) : src_(src) {}

void Program::add_header(std::string_view name, std::string_view src) {
  headers_.emplace_back(name, src);
}

void Program::add_compile_option(std::string_view option) {
  compile_options_.emplace_back(option);
}

void Program::instantiate_kernel(const std::string &name) {
  instantiate_names_.push_back(name);
}

Kernel Program::get_kernel_extern_c(std::string_view name) {
  if (!module_) {
    compile();
  }
  return Kernel(module_, name);
}

Kernel Program::get_kernel(const std::string &name) {
  if (!module_) {
    compile();
  }
  const std::string &lowered = lowered_names_.at(name);
  return get_kernel_extern_c(lowered);
}

std::string Program::read_file(std::string_view path) {
  std::ifstream f(path.data());
  std::string src(std::istreambuf_iterator<char>(f), {});
  return src;
}

void Program::add_optimization_flags() {
  const char *c = CMAKE_HIP_FLAGS;
  while (*c) {
    if (*c == ' ') {
      ++c;
      continue;
    }
    const char *start = c;
    while (*c && *c != ' ' && *c != '\t') {
      ++c;
    }
    ptrdiff_t len = c - start;
    const std::string_view option(start, len);
    add_compile_option(option);
  }
}

void Program::add_compatibility_headers() {
  const std::string path =
      std::string(DATA_DIR) + "/hip/include/api_compat.hpp";
  const std::string src = read_file(path);
  add_header("api_compat.hpp", src);
  add_compile_option("-I" + std::string(DATA_DIR) + "/hip/include");
}

void Program::add_numeric_headers() {
  for (std::string_view header : numeric_headers_) {
    const std::string path =
        std::string(NUMERIC_INCLUDE_DIR) + "/" + header.data();
    const std::string src = read_file(path);
    add_header(header, src);
  }
  add_compile_option("-I" + std::string(NUMERIC_INCLUDE_DIR) +
                     "/numeric/memory");
}

void Program::compile() {
  add_compatibility_headers();
  add_numeric_headers();
  add_optimization_flags();
  src_ = "#include <api_compat.hpp>\n\n" + src_;
  add_compile_option("-D__HIPCC_RTC__");
  add_compile_option("-D__HIP_DEVICE_COMPILE__");
  add_compile_option("-DNUMERIC_ENABLE_HIP=1");

  std::vector<const char *> header_sources;
  std::vector<const char *> header_names;
  for (const auto &[name, src] : headers_) {
    header_sources.push_back(src.data());
    header_names.push_back(name.data());
  }
  hiprtcProgram program;
  NUMERIC_CHECK_HIP(hiprtcCreateProgram(&program, src_.c_str(), "kernel.cu",
                                        headers_.size(), header_sources.data(),
                                        header_names.data()));
  for (const std::string &name : instantiate_names_) {
    NUMERIC_CHECK_HIP(hiprtcAddNameExpression(program, name.c_str()));
  }
  std::vector<const char *> options;
  for (const auto &option : compile_options_) {
    options.push_back(option.c_str());
  }
  if (const hiprtcResult status =
          hiprtcCompileProgram(program, options.size(), options.data());
      status != HIPRTC_SUCCESS) {
    size_t log_size;
    NUMERIC_CHECK_HIP(hiprtcGetProgramLogSize(program, &log_size));
    std::string log(log_size, '\0');
    NUMERIC_CHECK_HIP(hiprtcGetProgramLog(program, &log[0]));
    std::cerr << log << std::endl;
    NUMERIC_CHECK_HIP(status);
  }
  for (const std::string &name : instantiate_names_) {
    const char *lowered;
    NUMERIC_CHECK_HIP(hiprtcGetLoweredName(program, name.c_str(), &lowered));
    lowered_names_[name] = std::string(lowered);
  }
  size_t code_size;
  NUMERIC_CHECK_HIP(hiprtcGetCodeSize(program, &code_size));
  std::vector<char> binary(code_size);
  NUMERIC_CHECK_HIP(hiprtcGetCode(program, binary.data()));
  hiprtcDestroyProgram(&program);
  module_ = std::make_shared<Module>(binary);
}

} // namespace numeric::hip
