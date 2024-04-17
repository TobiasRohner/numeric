#ifndef NUMERIC_HIP_PROGRAM_HPP_
#define NUMERIC_HIP_PROGRAM_HPP_

#include <map>
#include <memory>
#include <numeric/hip/kernel.hpp>
#include <numeric/hip/module.hpp>
#include <numeric/hip/runtime.hpp>
#include <numeric/utils/type_name.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace numeric::hip {

namespace detail {

template <typename... Ts> struct ConvertTemplatePackDummy {};

} // namespace detail

/**
 * @brief Class representing a HIP program.
 */
class Program {
  struct Header {
    /**
     * @brief Constructs a Header object with the provided name and source.
     *
     * @param name Name of the header.
     * @param src Source code of the header.
     */
    Header(std::string_view name_, std::string_view src_);
    std::string name; /**< Name of the header. */
    std::string src;  /**< Source code of the header. */
  };

public:
  /**
   * @brief Constructs a Program object with the provided source code.
   *
   * @param src Source code of the program.
   */
  Program(std::string src);

  Program(const Program &) = default;
  Program(Program &&) = default;
  Program &operator=(const Program &) = default;
  Program &operator=(Program &&) = default;

  /**
   * @brief Adds a header to the program.
   *
   * @param name Name of the header.
   * @param src Source code of the header.
   */
  void add_header(std::string_view name, std::string_view src);

  /**
   * @brief Adds a compile option to the program.
   *
   * @param option Compile option to add.
   */
  void add_compile_option(std::string_view option);

  /**
   * @brief Instantiates a kernel with the given name.
   *
   * @param name Name of the kernel to instantiate.
   */
  void instantiate_kernel(const std::string &name);

  /**
   * @brief Instantiates a templated kernel with the given name and types.
   *
   * @tparam Types Types to instantiate the kernel with.
   * @param name Name of the kernel to instantiate.
   */
  template <typename... Types>
  void instantiate_kernel(const std::string &name) {
    const std::string name_with_type =
        name + std::string(types_to_template_pack<Types...>());
    instantiate_kernel(name_with_type);
  }

  /**
   * @brief Gets a kernel from the program with extern "C" linkage.
   *
   * @param name Name of the kernel. (Mangled if the kernel has C++ linkage)
   * @return Kernel object.
   */
  Kernel get_kernel_extern_c(std::string_view name);

  /**
   * @brief Gets a kernel from the program.
   *
   * @param name Name of the kernel.
   * @return Kernel object.
   */
  Kernel get_kernel(const std::string &name);
  template <typename... Types>
  [[nodiscard]] Kernel get_kernel(const std::string &name) {
    const std::string name_with_type =
        name + std::string(types_to_template_pack<Types...>());
    return get_kernel(name_with_type);
  }

private:
  std::string src_;
  std::vector<Header> headers_;
  std::vector<std::string> compile_options_;
  std::vector<std::string> instantiate_names_;
  std::map<std::string, std::string> lowered_names_;
  std::shared_ptr<Module> module_;
  static std::vector<std::string_view> numeric_headers_;

  static std::string read_file(std::string_view path);
  void add_optimization_flags();
  void add_compatibility_headers();
  void add_numeric_headers();
  void compile();

  template <typename... Types>
  static constexpr std::string_view types_to_template_pack() {
    using pack = detail::ConvertTemplatePackDummy<Types...>;
    const std::string_view name = utils::type_name<pack>();
    const size_t start_idx = name.find("<");
    return name.substr(start_idx);
  }
};

} // namespace numeric::hip

#endif
