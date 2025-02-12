#include <numeric/io/hdf5_dataspace.hpp>
#include <numeric/io/hdf5_error.hpp>
#include <numeric/io/hdf5_group.hpp>
#include <numeric/io/hdf5_type.hpp>
#include <numeric/io/hdf5_variable.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::io {

HDF5Group::~HDF5Group() {
  if (is_root_file()) {
    H5Fclose(id_);
  } else {
    H5Gclose(id_);
  }
}

HDF5Group::HDF5Group(hid_t id,
                     const std::shared_ptr<const HDF5Group> &root_file)
    : id_(id), root_file_(root_file) {}

extern "C" {
static herr_t push_back_variables(hid_t id, const char *name,
                                  const H5L_info_t *, void *op_data) {
  std::vector<std::string> *variable_names =
      static_cast<std::vector<std::string> *>(op_data);
  H5O_info_t info;
  const herr_t status =
      H5Oget_info_by_name(id, name, &info, H5O_INFO_BASIC, H5P_DEFAULT);
  if (status < 0) {
    return status;
  }
  if (info.type == H5O_TYPE_DATASET) {
    variable_names->emplace_back(name);
  }
  return status;
}
}

std::vector<std::string> HDF5Group::do_get_variable_names() const {
  std::vector<std::string> variable_names;
  const herr_t status = H5Literate(id_, H5_INDEX_NAME, H5_ITER_NATIVE, NULL,
                                   push_back_variables, &variable_names);
  if (status < 0) {
    NUMERIC_ERROR("Failed to get variable names\n\n{}", get_hdf5_stacktrace());
  }
  return variable_names;
}

extern "C" {
static herr_t push_back_groups(hid_t id, const char *name, const H5L_info_t *,
                               void *op_data) {
  std::vector<std::string> *group_names =
      static_cast<std::vector<std::string> *>(op_data);
  H5O_info_t info;
  const herr_t status =
      H5Oget_info_by_name(id, name, &info, H5O_INFO_BASIC, H5P_DEFAULT);
  if (status < 0) {
    return status;
  }
  if (info.type == H5O_TYPE_GROUP) {
    group_names->emplace_back(name);
  }
  return status;
}
}

std::vector<std::string> HDF5Group::do_get_group_names() const {
  std::vector<std::string> group_names;
  const herr_t status = H5Literate(id_, H5_INDEX_NAME, H5_ITER_NATIVE, NULL,
                                   push_back_groups, &group_names);
  if (status < 0) {
    NUMERIC_ERROR("Failed to get group names\n\n{}", get_hdf5_stacktrace());
  }
  return group_names;
}

bool HDF5Group::do_variable_exists(std::string_view name) const {
  if (!link_exists(name)) {
    return false;
  }
  const H5O_type_t type = get_object_info(name).type;
  return type == H5O_TYPE_DATASET;
}

bool HDF5Group::do_group_exists(std::string_view name) const {
  if (!link_exists(name)) {
    return false;
  }
  const H5O_type_t type = get_object_info(name).type;
  return type == H5O_TYPE_GROUP;
}

std::shared_ptr<HierarchicalFile>
HDF5Group::do_open_group(std::string_view name) {
  const hid_t id = H5Gopen(id_, name.data(), H5P_DEFAULT);
  if (id == H5I_INVALID_HID) {
    NUMERIC_ERROR("Failed to open group\n\n{}", get_hdf5_stacktrace());
  }
  return std::shared_ptr<HDF5Group>(new HDF5Group(id, root_file()));
}

std::shared_ptr<const HierarchicalFile>
HDF5Group::do_open_group(std::string_view name) const {
  const hid_t id = H5Gopen(id_, name.data(), H5P_DEFAULT);
  if (id == H5I_INVALID_HID) {
    NUMERIC_ERROR("Failed to open group\n\n{}", get_hdf5_stacktrace());
  }
  return std::shared_ptr<const HDF5Group>(new HDF5Group(id, root_file()));
}

std::shared_ptr<HierarchicalFile>
HDF5Group::do_create_group(std::string_view name) {
  const hid_t id =
      H5Gcreate(id_, name.data(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (id == H5I_INVALID_HID) {
    NUMERIC_ERROR("Failed to create new group\n\n{}", get_hdf5_stacktrace());
  }
  return std::shared_ptr<HDF5Group>(new HDF5Group(id, root_file()));
}

std::shared_ptr<Variable> HDF5Group::do_open_variable(std::string_view name) {
  const hid_t id = H5Dopen(id_, name.data(), H5P_DEFAULT);
  if (id == H5I_INVALID_HID) {
    NUMERIC_ERROR("Unable to open variable\n\n{}", get_hdf5_stacktrace());
  }
  return std::make_shared<HDF5Variable>(id, root_file());
}

std::shared_ptr<const Variable>
HDF5Group::do_open_variable(std::string_view name) const {
  const hid_t id = H5Dopen(id_, name.data(), H5P_DEFAULT);
  if (id == H5I_INVALID_HID) {
    NUMERIC_ERROR("Unable to open variable\n\n{}", get_hdf5_stacktrace());
  }
  return std::make_shared<const HDF5Variable>(id, root_file());
}

std::shared_ptr<Variable> HDF5Group::do_create_variable(std::string_view name,
                                                        utils::Datatype type,
                                                        dim_t ndims,
                                                        const dim_t *dims) {
  std::vector<hsize_t> dsdims(ndims);
  for (size_t i = 0; i < ndims; ++i) {
    dsdims[i] = static_cast<hsize_t>(dims[i]);
  }
  HDF5Dataspace dataspace(ndims, dsdims.data());
  HDF5Type h5t(type);
  const hid_t id = H5Dcreate(id_, name.data(), static_cast<hid_t>(h5t),
                             static_cast<hid_t>(dataspace), H5P_DEFAULT,
                             H5P_DEFAULT, H5P_DEFAULT);
  if (id == H5I_INVALID_HID) {
    NUMERIC_ERROR("Failed to create dataset\n\n{}", get_hdf5_stacktrace());
  }
  return std::make_shared<HDF5Variable>(id, root_file());
}

bool HDF5Group::is_root_file() const { return root_file_ == nullptr; }

std::shared_ptr<const HDF5Group> HDF5Group::root_file() const {
  if (is_root_file()) {
    return this->shared_from_this();
  } else {
    return root_file_;
  }
}

bool HDF5Group::link_exists(std::string_view name) const {
  const htri_t err = H5Lexists(id_, name.data(), H5P_DEFAULT);
  if (err < 0) {
    NUMERIC_ERROR("Failed to test existence of link\n\n{}",
                  get_hdf5_stacktrace());
  }
  return err > 0;
}

H5O_info2_t HDF5Group::get_object_info(std::string_view name) const {
  H5O_info2_t info;
  const herr_t err =
      H5Oget_info_by_name(id_, name.data(), &info, H5O_INFO_BASIC, H5P_DEFAULT);
  if (err < 0) {
    NUMERIC_ERROR("Unable to get object information\n\n{}",
                  get_hdf5_stacktrace());
  }
  return info;
}

} // namespace numeric::io
