#include <netcdf.h>
#include <numeric/io/netcdf_error.hpp>
#include <numeric/io/netcdf_file.hpp>
#include <numeric/io/netcdf_types.hpp>
#include <numeric/io/netcdf_variable.hpp>

namespace numeric::io {

NetCDFFile::~NetCDFFile() {
  if (is_root_file()) {
    nc_close(ncid_);
  }
}

std::shared_ptr<NetCDFFile> NetCDFFile::open(std::string_view path,
                                             FileMode mode) {
  int ncid;
  NUMERIC_CHECK_NETCDF_STATUS(
      nc_open(path.data(), file_mode_to_mode(mode), &ncid));
  return std::shared_ptr<NetCDFFile>(new NetCDFFile(ncid, nullptr));
}

std::shared_ptr<NetCDFFile> NetCDFFile::create(std::string_view path,
                                               FileMode mode) {
  int ncid;
  NUMERIC_CHECK_NETCDF_STATUS(
      nc_create(path.data(), file_mode_to_mode(mode) | NC_NETCDF4, &ncid));
  return std::shared_ptr<NetCDFFile>(new NetCDFFile(ncid, nullptr));
}

std::vector<NetCDFDimension> NetCDFFile::get_dims() const {
  const int ndims = get_num_dims();
  std::vector<NetCDFDimension> dims;
  for (int dimid = 0; dimid < ndims; ++dimid) {
    dims.push_back(NetCDFDimension(ncid_, dimid, get_root_file()));
  }
  return dims;
}

NetCDFDimension NetCDFFile::get_dim(std::string_view name) const {
  int dimid;
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_dimid(ncid_, name.data(), &dimid));
  return NetCDFDimension(ncid_, dimid, get_root_file());
}

NetCDFDimension NetCDFFile::create_dim(std::string_view name, size_t len) {
  int dimid;
  NUMERIC_CHECK_NETCDF_STATUS(nc_def_dim(ncid_, name.data(), len, &dimid));
  return NetCDFDimension(ncid_, dimid, get_root_file());
}

std::shared_ptr<NetCDFFile> NetCDFFile::open_group(std::string_view name) {
  std::shared_ptr<NetCDFFile> group =
      dynamic_pointer_cast<NetCDFFile>(HierarchicalFile::open_group(name));
  if (!group) {
    NUMERIC_ERROR(
        "This should not have happened. Congratulations for breaking the code");
  }
  return group;
}

std::shared_ptr<const NetCDFFile>
NetCDFFile::open_group(std::string_view name) const {
  std::shared_ptr<const NetCDFFile> group =
      dynamic_pointer_cast<const NetCDFFile>(
          HierarchicalFile::open_group(name));
  if (!group) {
    NUMERIC_ERROR(
        "This should not have happened. Congratulations for breaking the code");
  }
  return group;
}

std::shared_ptr<NetCDFFile> NetCDFFile::create_group(std::string_view name) {
  std::shared_ptr<NetCDFFile> group =
      dynamic_pointer_cast<NetCDFFile>(HierarchicalFile::create_group(name));
  if (!group) {
    NUMERIC_ERROR(
        "This should not have happened. Congratulations for breaking the code");
  }
  return group;
}

std::shared_ptr<NetCDFVariable>
NetCDFFile::open_variable(std::string_view name) {
  std::shared_ptr<NetCDFVariable> variable =
      dynamic_pointer_cast<NetCDFVariable>(
          HierarchicalFile::open_variable(name));
  if (!variable) {
    NUMERIC_ERROR(
        "This should not have happened. Congratulations for breaking the code");
  }
  return variable;
}

std::shared_ptr<const NetCDFVariable>
NetCDFFile::open_variable(std::string_view name) const {
  std::shared_ptr<const NetCDFVariable> variable =
      dynamic_pointer_cast<const NetCDFVariable>(
          HierarchicalFile::open_variable(name));
  if (!variable) {
    NUMERIC_ERROR(
        "This should not have happened. Congratulations for breaking the code");
  }
  return variable;
}

NetCDFFile::NetCDFFile(int ncid,
                       const std::shared_ptr<const NetCDFFile> &root_file)
    : root_file_(root_file), ncid_(ncid) {}

std::vector<std::string> NetCDFFile::do_get_variable_names() const {
  std::vector<std::string> varnames;
  const std::vector<int> varids = get_variable_ids();
  for (int varid : varids) {
    varnames.emplace_back(get_variable_name(varid));
  }
  return varnames;
}

std::vector<std::string> NetCDFFile::do_get_group_names() const {
  std::vector<std::string> grpnames;
  const std::vector<int> grpids = get_group_ids();
  for (int grpid : grpids) {
    grpnames.emplace_back(get_group_name(grpid));
  }
  return grpnames;
}

bool NetCDFFile::do_variable_exists(std::string_view name) const {
  return varname_exists(name);
}

bool NetCDFFile::do_group_exists(std::string_view name) const {
  return grpname_exists(name);
}

std::shared_ptr<Variable> NetCDFFile::do_open_variable(std::string_view name) {
  int varid;
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_varid(ncid_, name.data(), &varid));
  return std::make_shared<NetCDFVariable>(ncid_, varid, get_root_file());
}

std::shared_ptr<const Variable>
NetCDFFile::do_open_variable(std::string_view name) const {
  int varid;
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_varid(ncid_, name.data(), &varid));
  return std::make_shared<const NetCDFVariable>(ncid_, varid, get_root_file());
}

std::shared_ptr<Variable> NetCDFFile::do_create_variable(std::string_view name,
                                                         utils::Datatype type,
                                                         dim_t ndims,
                                                         const dim_t *dims) {
  std::vector<NetCDFDimension> ncdims;
  for (dim_t i = 0; i < ndims; ++i) {
    const std::string dimname =
        next_available_dimname(std::string(name) + "_dim");
    ncdims.push_back(create_dim(dimname, dims[i]));
  }
  return do_create_variable(name, type, ncdims);
}

std::shared_ptr<NetCDFVariable>
NetCDFFile::do_create_variable(std::string_view name, utils::Datatype type,
                               const std::vector<NetCDFDimension> &dims) {
  std::vector<int> dimids;
  for (const auto &dim : dims) {
    dimids.push_back(dim.dimid_);
  }
  const nc_type xtype = datatype_to_netcdf_type(type);
  int varid;
  NUMERIC_CHECK_NETCDF_STATUS(nc_def_var(ncid_, name.data(), xtype,
                                         dimids.size(), dimids.data(), &varid));
  return std::make_shared<NetCDFVariable>(ncid_, varid, get_root_file());
}

std::shared_ptr<HierarchicalFile>
NetCDFFile::do_open_group(std::string_view name) {
  const int grpid = get_group_id(name);
  return std::shared_ptr<NetCDFFile>(new NetCDFFile(grpid, get_root_file()));
}

std::shared_ptr<const HierarchicalFile>
NetCDFFile::do_open_group(std::string_view name) const {
  const int grpid = get_group_id(name);
  return std::shared_ptr<NetCDFFile>(new NetCDFFile(grpid, get_root_file()));
}

std::shared_ptr<HierarchicalFile>
NetCDFFile::do_create_group(std::string_view name) {
  int grpid;
  NUMERIC_CHECK_NETCDF_STATUS(nc_def_grp(ncid_, name.data(), &grpid));
  return std::shared_ptr<NetCDFFile>(new NetCDFFile(grpid, get_root_file()));
}

int NetCDFFile::file_mode_to_mode(FileMode mode) {
  switch (mode) {
  case FileMode::READ:
    return NC_NOWRITE;
  case FileMode::WRITE:
    return NC_WRITE;
  case FileMode::KEEP_EXISTING:
    return NC_NOCLOBBER;
  case FileMode::OVERWRITE:
    return NC_CLOBBER;
  default:
    return -1;
  }
}

bool NetCDFFile::is_root_file() const { return root_file_ == nullptr; }

std::shared_ptr<const NetCDFFile> NetCDFFile::get_root_file() const {
  std::shared_ptr<const NetCDFFile> root = root_file_;
  if (root == nullptr) {
    root = this->shared_from_this();
  }
  return root;
}

int NetCDFFile::get_num_dims() const {
  int ndims;
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_ndims(ncid_, &ndims));
  return ndims;
}

int NetCDFFile::get_num_variables() const {
  int nvars;
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_varids(ncid_, &nvars, NULL));
  return nvars;
}

std::vector<int> NetCDFFile::get_variable_ids() const {
  const int nvars = get_num_variables();
  std::vector<int> varids(nvars);
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_varids(ncid_, NULL, varids.data()));
  varids.resize(nvars);
  return varids;
}

int NetCDFFile::get_variable_id(std::string_view name) const {
  int varid;
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_varid(ncid_, name.data(), &varid));
  return varid;
}

std::string NetCDFFile::get_variable_name(int varid) const {
  char name[NC_MAX_NAME];
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_varname(ncid_, varid, name));
  return std::string(name);
}

int NetCDFFile::get_num_groups() const {
  int ngrps;
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_grps(ncid_, &ngrps, NULL));
  return ngrps;
}

std::vector<int> NetCDFFile::get_group_ids() const {
  const int ngrps = get_num_groups();
  std::vector<int> grpids(ngrps);
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_grps(ncid_, NULL, grpids.data()));
  return grpids;
}

int NetCDFFile::get_group_id(std::string_view name) const {
  int grpid;
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_grp_ncid(ncid_, name.data(), &grpid));
  return grpid;
}

std::string NetCDFFile::get_group_name(int grpid) const {
  size_t len;
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_grpname_len(grpid, &len));
  auto name = std::make_unique<char[]>(len);
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_grpname(grpid, name.get()));
  return std::string(name.get());
}

bool NetCDFFile::dimname_exists(std::string_view name) const {
  int dimid;
  const int status = nc_inq_dimid(ncid_, name.data(), &dimid);
  if (status == NC_NOERR) {
    return true;
  } else if (status == NC_EBADDIM) {
    return false;
  } else {
    NUMERIC_CHECK_NETCDF_STATUS(status);
    NUMERIC_ERROR("This should never be reached");
  }
}

std::string
NetCDFFile::next_available_dimname(std::string_view basename) const {
  std::string dimname;
  size_t idx = 0;
  do {
    dimname = std::string(basename) + '_' + std::to_string(idx);
    ++idx;
  } while (dimname_exists(dimname));
  return dimname;
}

bool NetCDFFile::varname_exists(std::string_view name) const {
  int varid;
  const int status = nc_inq_varid(ncid_, name.data(), &varid);
  if (status == NC_NOERR) {
    return true;
  } else if (status == NC_ENOTVAR) {
    return false;
  } else {
    NUMERIC_CHECK_NETCDF_STATUS(status);
    NUMERIC_ERROR("This should never be reached");
  }
}

bool NetCDFFile::grpname_exists(std::string_view name) const {
  int grpid;
  const int status = nc_inq_grp_ncid(ncid_, name.data(), &grpid);
  if (status == NC_NOERR) {
    return true;
  } else if (status == NC_ENOGRP) {
    return false;
  } else {
    NUMERIC_CHECK_NETCDF_STATUS(status);
    NUMERIC_ERROR("This should never be reached");
  }
}

} // namespace numeric::io
