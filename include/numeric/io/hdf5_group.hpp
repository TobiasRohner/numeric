#ifndef NUMERIC_IO_HDF5_GROUP_HPP_
#define NUMERIC_IO_HDF5_GROUP_HPP_

#include <hdf5.h>
#include <memory>
#include <numeric/io/hierarchical_file.hpp>

namespace numeric::io {

class HDF5Group : public HierarchicalFile,
                  public std::enable_shared_from_this<HDF5Group> {
  using super = HierarchicalFile;

public:
  HDF5Group(const HDF5Group &) = delete;
  HDF5Group(HDF5Group &&) = delete;
  HDF5Group &operator=(const HDF5Group &) = delete;
  HDF5Group &operator=(HDF5Group &&) = delete;
  virtual ~HDF5Group() override;

protected:
  hid_t id_;
  std::shared_ptr<const HDF5Group> root_file_;

  HDF5Group(hid_t id, const std::shared_ptr<const HDF5Group> &root_file);

  virtual std::vector<std::string> do_get_variable_names() const override;

  virtual std::vector<std::string> do_get_group_names() const override;

  virtual bool do_variable_exists(std::string_view name) const override;

  virtual bool do_group_exists(std::string_view name) const override;

  virtual std::shared_ptr<HierarchicalFile>
  do_open_group(std::string_view name) override;

  virtual std::shared_ptr<const HierarchicalFile>
  do_open_group(std::string_view name) const override;

  virtual std::shared_ptr<HierarchicalFile>
  do_create_group(std::string_view name) override;

  virtual std::shared_ptr<Variable>
  do_open_variable(std::string_view name) override;

  virtual std::shared_ptr<const Variable>
  do_open_variable(std::string_view name) const override;

  virtual std::shared_ptr<Variable>
  do_create_variable(std::string_view name, utils::Datatype type, dim_t ndims,
                     const dim_t *dims) override;

private:
  bool is_root_file() const;
  std::shared_ptr<const HDF5Group> root_file() const;
  bool link_exists(std::string_view name) const;
  H5O_info2_t get_object_info(std::string_view name) const;
};

} // namespace numeric::io

#endif
