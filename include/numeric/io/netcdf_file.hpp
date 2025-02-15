#ifndef NUMERIC_IO_NETCDF_FILE_HPP_
#define NUMERIC_IO_NETCDF_FILE_HPP_

#include <memory>
#include <numeric/io/file_mode.hpp>
#include <numeric/io/hierarchical_file.hpp>
#include <numeric/io/netcdf_dimension.hpp>
#include <numeric/io/netcdf_variable.hpp>

namespace numeric::io {

class NetCDFFile : public HierarchicalFile,
                   public std::enable_shared_from_this<NetCDFFile> {
  using super = HierarchicalFile;

public:
  NetCDFFile(const NetCDFFile &) = delete;
  NetCDFFile(NetCDFFile &&) = delete;
  NetCDFFile &operator=(const NetCDFFile &) = delete;
  NetCDFFile &operator=(NetCDFFile &&) = delete;
  virtual ~NetCDFFile() override;

  static std::shared_ptr<NetCDFFile> open(std::string_view path,
                                          FileMode mode = FileMode::READ);
  static std::shared_ptr<NetCDFFile>
  create(std::string_view path, FileMode mode = FileMode::OVERWRITE);

  std::vector<NetCDFDimension> get_dims() const;
  NetCDFDimension get_dim(std::string_view name) const;
  NetCDFDimension create_dim(std::string_view name, size_t len);

  std::shared_ptr<NetCDFFile> open_group(std::string_view name);
  std::shared_ptr<const NetCDFFile> open_group(std::string_view name) const;
  std::shared_ptr<NetCDFFile> create_group(std::string_view name);

  std::shared_ptr<NetCDFVariable> open_variable(std::string_view name);
  std::shared_ptr<const NetCDFVariable>
  open_variable(std::string_view name) const;
  template <typename T, dim_t N>
  std::shared_ptr<NetCDFVariable>
  create_variable(std::string_view name, const memory::Shape<N> &shape) {
    std::shared_ptr<NetCDFVariable> variable =
        dynamic_pointer_cast<NetCDFVariable>(
            HierarchicalFile::create_variable<T>(name, shape));
    if (!variable) {
      NUMERIC_ERROR("This should not have happened. Congratulations for "
                    "breaking the code");
    }
    return variable;
  }
  template <typename T>
  std::shared_ptr<NetCDFVariable>
  create_variable(std::string_view name,
                  const std::vector<NetCDFDimension> &dims) {
    return do_create_variable(name, utils::to_datatype_v<T>, dims);
  }
  using super::create_variable;

  template <typename T, dim_t N>
  void write(std::string_view name, const memory::ArrayConstView<T, N> &arr,
             const std::vector<NetCDFDimension> &dims) {
    std::shared_ptr<Variable> var;
    if (variable_exists(name)) {
      var = open_variable(name);
    } else {
      var = create_variable<T>(name, dims);
    }
    var->write(arr);
  }
  using super::write;

protected:
  std::shared_ptr<const NetCDFFile> root_file_;
  int ncid_;

  NetCDFFile(int ncid, const std::shared_ptr<const NetCDFFile> &root_file);

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
  std::shared_ptr<NetCDFVariable>
  do_create_variable(std::string_view name, utils::Datatype type,
                     const std::vector<NetCDFDimension> &dims);

private:
  static int file_mode_to_mode(FileMode mode);
  bool is_root_file() const;
  std::shared_ptr<const NetCDFFile> get_root_file() const;
  int get_num_dims() const;
  int get_num_variables() const;
  std::vector<int> get_variable_ids() const;
  int get_variable_id(std::string_view name) const;
  std::string get_variable_name(int varid) const;
  int get_num_groups() const;
  std::vector<int> get_group_ids() const;
  int get_group_id(std::string_view name) const;
  std::string get_group_name(int grpid) const;
  bool dimname_exists(std::string_view name) const;
  std::string next_available_dimname(std::string_view basename) const;
  bool varname_exists(std::string_view name) const;
  bool grpname_exists(std::string_view name) const;
};

} // namespace numeric::io

#endif
