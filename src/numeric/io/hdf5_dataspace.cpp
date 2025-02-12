#include <memory>
#include <numeric/io/hdf5_dataspace.hpp>
#include <numeric/io/hdf5_error.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::io {

HDF5Dataspace::HDF5Dataspace() {
  id_ = H5Screate(H5S_SCALAR);
  if (id_ == H5I_INVALID_HID) {
    NUMERIC_ERROR("Unable to create dataspace\n\n{}", get_hdf5_stacktrace());
  }
}

HDF5Dataspace::HDF5Dataspace(hid_t id) : id_(id) {}

HDF5Dataspace::HDF5Dataspace(int ndims, const hsize_t *dims) {
  id_ = H5Screate_simple(ndims, dims, nullptr);
  if (id_ == H5I_INVALID_HID) {
    NUMERIC_ERROR("Unable to create dataspace\n\n{}", get_hdf5_stacktrace());
  }
}

HDF5Dataspace::HDF5Dataspace(HDF5Dataspace &&other) : id_(other.id_) {
  other.id_ = H5I_INVALID_HID;
}

HDF5Dataspace::~HDF5Dataspace() {
  if (id_ != H5I_INVALID_HID) {
    H5Sclose(id_);
  }
}

HDF5Dataspace &HDF5Dataspace::operator=(HDF5Dataspace &&other) {
  if (std::addressof(other) != this) {
    id_ = other.id_;
    other.id_ = H5I_INVALID_HID;
  }
  return *this;
}

HDF5Dataspace::operator hid_t() const { return id_; }

std::vector<dim_t> HDF5Dataspace::dims() const {
  const size_t ndims = num_dims();
  std::vector<hsize_t> dims(ndims);
  const int err = H5Sget_simple_extent_dims(id_, dims.data(), nullptr);
  if (err < 0) {
    NUMERIC_ERROR("Unable to get dimensions of dataspace\n\n{}",
                  get_hdf5_stacktrace());
  }
  std::vector<dim_t> dimsize(ndims);
  for (size_t i = 0; i < ndims; ++i) {
    dimsize[i] = static_cast<dim_t>(dims[i]);
  }
  return dimsize;
}

void HDF5Dataspace::select_hyperslab(const hsize_t *start,
                                     const hsize_t *stride,
                                     const hsize_t *count) {
  const std::vector<hsize_t> block(num_dims(), 1);
  const herr_t err = H5Sselect_hyperslab(id_, H5S_SELECT_SET, start, stride,
                                         count, block.data());
  if (err < 0) {
    NUMERIC_ERROR("Unable to select hyperslab\n\n{}", get_hdf5_stacktrace());
  }
}

size_t HDF5Dataspace::num_dims() const {
  const int ndims = H5Sget_simple_extent_ndims(id_);
  if (ndims < 0) {
    NUMERIC_ERROR("Unable to get number of dimensions\n\n{}",
                  get_hdf5_stacktrace());
  }
  return ndims;
}

} // namespace numeric::io
