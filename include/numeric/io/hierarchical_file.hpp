#ifndef NUMERIC_IO_HIERARCHICAL_FILE_HPP_
#define NUMERIC_IO_HIERARCHICAL_FILE_HPP_

#include <memory>
#include <numeric/config.hpp>
#include <numeric/io/variable.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_const_view.hpp>
#include <numeric/memory/array_view.hpp>
#include <numeric/memory/layout.hpp>
#include <numeric/utils/datatype.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace numeric::io {

class HierarchicalFile {
public:
  HierarchicalFile() = default;
  HierarchicalFile(const HierarchicalFile &) = default;
  HierarchicalFile(HierarchicalFile &&) = default;
  HierarchicalFile &operator=(const HierarchicalFile &) = default;
  HierarchicalFile &operator=(HierarchicalFile &&) = default;
  virtual ~HierarchicalFile() = default;

  std::vector<std::string> get_variable_names() const;
  std::vector<std::string> get_group_names() const;
  bool variable_exists(std::string_view name) const;
  bool group_exists(std::string_view name) const;

  std::shared_ptr<HierarchicalFile> open_group(std::string_view name);
  std::shared_ptr<const HierarchicalFile>
  open_group(std::string_view name) const;
  std::shared_ptr<HierarchicalFile> create_group(std::string_view name);

  std::shared_ptr<Variable> open_variable(std::string_view name);
  std::shared_ptr<const Variable> open_variable(std::string_view name) const;
  template <typename T, dim_t N>
  std::shared_ptr<Variable> create_variable(std::string_view name,
                                            const memory::Shape<N> &shape) {
    return do_create_variable(name, utils::to_datatype_v<T>, N, shape.raw());
  }
  template <typename T, dim_t N>
  void read(std::string_view name, memory::ArrayView<T, N> arr) const {
    const std::shared_ptr<const Variable> var = open_variable(name);
    var->read(arr);
  }
  template <typename T, dim_t N>
  memory::Array<T, N>
  read(std::string_view name,
       memory::MemoryType memory_type = memory::MemoryType::HOST) const {
    const std::shared_ptr<const Variable> var = open_variable(name);
    return var->read<T, N>(memory_type);
  }
  template <typename T, dim_t N>
  void write(std::string_view name, const memory::ArrayConstView<T, N> &arr) {
    std::shared_ptr<Variable> var;
    if (variable_exists(name)) {
      var = open_variable(name);
    } else {
      var = create_variable<T>(name, arr.shape());
    }
    var->write(arr);
  }

protected:
  virtual std::vector<std::string> do_get_variable_names() const = 0;
  virtual std::vector<std::string> do_get_group_names() const = 0;
  virtual bool do_variable_exists(std::string_view name) const = 0;
  virtual bool do_group_exists(std::string_view name) const = 0;
  virtual std::shared_ptr<HierarchicalFile>
  do_open_group(std::string_view name) = 0;
  virtual std::shared_ptr<const HierarchicalFile>
  do_open_group(std::string_view name) const = 0;
  virtual std::shared_ptr<HierarchicalFile>
  do_create_group(std::string_view name) = 0;
  virtual std::shared_ptr<Variable> do_open_variable(std::string_view name) = 0;
  virtual std::shared_ptr<const Variable>
  do_open_variable(std::string_view name) const = 0;
  virtual std::shared_ptr<Variable> do_create_variable(std::string_view name,
                                                       utils::Datatype type,
                                                       dim_t ndims,
                                                       const dim_t *dims) = 0;
};

} // namespace numeric::io

#endif
