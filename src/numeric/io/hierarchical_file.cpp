#include <numeric/io/hierarchical_file.hpp>

namespace numeric::io {

std::vector<std::string> HierarchicalFile::get_variable_names() const {
  return do_get_variable_names();
}

std::vector<std::string> HierarchicalFile::get_group_names() const {
  return do_get_group_names();
}

bool HierarchicalFile::variable_exists(std::string_view name) const {
  return do_variable_exists(name);
}

bool HierarchicalFile::group_exists(std::string_view name) const {
  return do_group_exists(name);
}

std::shared_ptr<const HierarchicalFile>
HierarchicalFile::open_group(std::string_view name) const {
  return do_open_group(name);
}

std::shared_ptr<HierarchicalFile>
HierarchicalFile::open_group(std::string_view name) {
  return do_open_group(name);
}

std::shared_ptr<HierarchicalFile>
HierarchicalFile::create_group(std::string_view name) {
  return do_create_group(name);
}

std::shared_ptr<const Variable>
HierarchicalFile::open_variable(std::string_view name) const {
  return do_open_variable(name);
}

std::shared_ptr<Variable>
HierarchicalFile::open_variable(std::string_view name) {
  return do_open_variable(name);
}

} // namespace numeric::io
