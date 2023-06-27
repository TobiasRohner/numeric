#ifndef NUMERIC_HIP_PROGRAM_HPP_
#define NUMERIC_HIP_PROGRAM_HPP_

#include <numeric/hip/module.hpp>
#include <numeric/hip/kernel.hpp>
#include <numeric/utils/type_name.hpp>
#include <memory>
#include <string_view>
#include <string>
#include <vector>
#include <map>
#include <hip/hiprtc.h>


namespace numeric::hip {

namespace detail {

template<typename... Ts>
struct ConvertTemplatePackDummy {};

}

class Program {
  struct Header {
    Header(std::string_view name_, std::string_view src_);
    std::string name;
    std::string src;
  };

public:
  Program(std::string src);
  Program(const Program &) = default;
  Program(Program &&) = default;
  Program &operator=(const Program &) = default;
  Program &operator=(Program &&) = default;

  void add_header(std::string_view name, std::string_view src);
  void add_compile_option(std::string_view option);

  void instantiate_kernel(const std::string &name);
  template<typename... Types> void instantiate_kernel(const std::string &name) {
    const std::string name_with_type = name + std::string(types_to_template_pack<Types...>());
    instantiate_kernel(name_with_type);
  }

  [[nodiscard]] Kernel get_kernel_extern_c(std::string_view name);
  [[nodiscard]] Kernel get_kernel(const std::string &name);
  template<typename... Types> [[nodiscard]] Kernel get_kernel(const std::string &name) {
    const std::string name_with_type = name + std::string(types_to_template_pack<Types...>());
    return get_kernel(name_with_type);
  }

private:
  std::string src_;
  std::vector<Header> headers_;
  std::vector<std::string> compile_options_;
  std::vector<std::string> instantiate_names_;
  std::map<std::string, std::string> lowered_names_;
  std::shared_ptr<Module> module_;

  void compile();

  template<typename... Types>
  static constexpr std::string_view types_to_template_pack() {
    using pack = detail::ConvertTemplatePackDummy<Types...>;
    const std::string_view name = utils::type_name<pack>();
    const size_t start_idx = name.find("<");
    return name.substr(start_idx);
  }
};

}


#endif
