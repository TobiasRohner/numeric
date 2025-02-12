#include <numeric/io/hdf5_error.hpp>
#include <numeric/io/hdf5_variable.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::io {

HDF5Variable::HDF5Variable(hid_t id,
                           const std::shared_ptr<const HDF5Group> &root_file)
    : id_(id), root_file_(root_file) {}

HDF5Variable::~HDF5Variable() { H5Dclose(id_); }

utils::Datatype HDF5Variable::do_datatype() const {
  const HDF5Type type = get_datatype();
  if (type.is_float() || type.is_integer()) {
    return static_cast<utils::Datatype>(type);
  }
  if (type.is_array()) {
    const HDF5Type underlying_type = type.super();
    return static_cast<utils::Datatype>(underlying_type);
  }
  NUMERIC_ERROR("Unsupported Datatype");
}

std::vector<dim_t> HDF5Variable::do_dims() const {
  return get_dataspace().dims();
}

void HDF5Variable::do_read(void *data, const dim_t *shape, const dim_t *stride,
                           dim_t N) const {
  HDF5Dataspace dataspace = get_dataspace();
  std::vector<hsize_t> argstart(N);
  std::vector<hsize_t> argstride(N);
  std::vector<hsize_t> argcount(N);
  for (dim_t i = 0; i < N; ++i) {
    argstart[i] = 0;
    argstride[i] = stride[i];
    argcount[i] = shape[i];
  }
  dataspace.select_hyperslab(argstart.data(), argstride.data(),
                             argcount.data());
  HDF5Dataspace memspace(N, argcount.data());
  const HDF5Type type = get_datatype();
  const herr_t err =
      H5Dread(id_, static_cast<hid_t>(type), static_cast<hid_t>(memspace),
              static_cast<hid_t>(dataspace), H5P_DEFAULT, data);
  if (err < 0) {
    NUMERIC_ERROR("Failed to read from dataset\n\n{}", get_hdf5_stacktrace());
  }
}

void HDF5Variable::do_write(const void *data, const dim_t *shape,
                            const dim_t *stride, dim_t N) {
  HDF5Dataspace dataspace = get_dataspace();
  std::vector<hsize_t> argstart(N);
  std::vector<hsize_t> argstride(N);
  std::vector<hsize_t> argcount(N);
  dim_t slicesize = 1;
  for (dim_t i = 0; i < N; ++i) {
    slicesize *= shape[i];
  }
  for (dim_t i = 0; i < N; ++i) {
    slicesize /= shape[i];
    argstart[i] = 0;
    argstride[i] = stride[i] / slicesize;
    argcount[i] = shape[i];
  }
  dataspace.select_hyperslab(argstart.data(), argstride.data(),
                             argcount.data());
  HDF5Dataspace memspace(N, argcount.data());
  const HDF5Type type = get_datatype();
  const herr_t err =
      H5Dwrite(id_, static_cast<hid_t>(type), static_cast<hid_t>(memspace),
               static_cast<hid_t>(dataspace), H5P_DEFAULT, data);
  if (err < 0) {
    NUMERIC_ERROR("Failed to write to dataset\n\n{}", get_hdf5_stacktrace());
  }
}

HDF5Type HDF5Variable::get_datatype() const {
  const hid_t id = H5Dget_type(id_);
  if (id == H5I_INVALID_HID) {
    NUMERIC_ERROR("Unable to get datatype for variable\n\n{}",
                  get_hdf5_stacktrace());
  }
  return HDF5Type(id);
}

HDF5Dataspace HDF5Variable::get_dataspace() const {
  const hid_t id = H5Dget_space(id_);
  if (id == H5I_INVALID_HID) {
    NUMERIC_ERROR("Unable to get dataspace of variable\n\n{}",
                  get_hdf5_stacktrace());
  }
  return HDF5Dataspace(id);
}

} // namespace numeric::io
