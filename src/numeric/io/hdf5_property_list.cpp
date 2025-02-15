#include <numeric/io/hdf5_error.hpp>
#include <numeric/io/hdf5_property_list.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::io {

HDF5PropertyList::HDF5PropertyList() : id_(H5I_INVALID_HID) {}

HDF5PropertyList::HDF5PropertyList(hid_t cls_id) : HDF5PropertyList() {
  id_ = H5Pcreate(cls_id);
  if (id_ == H5I_INVALID_HID) {
    NUMERIC_ERROR("Failed to create property list\n\n{}",
                  get_hdf5_stacktrace());
  }
}

HDF5PropertyList::HDF5PropertyList(const HDF5PropertyList &other)
    : HDF5PropertyList() {
  id_ = H5Pcopy(static_cast<hid_t>(other));
  if (id_ == H5I_INVALID_HID) {
    NUMERIC_ERROR("Failed to copy property list\n\n{}", get_hdf5_stacktrace());
  }
}

HDF5PropertyList::HDF5PropertyList(HDF5PropertyList &&other) {
  id_ = other.id_;
  other.id_ = H5I_INVALID_HID;
}

HDF5PropertyList::~HDF5PropertyList() {
  if (id_ != H5I_INVALID_HID) {
    const herr_t err = H5Pclose(id_);
    if (err < 0) {
      NUMERIC_ERROR("Failed to close property list\n\n{}",
                    get_hdf5_stacktrace());
    }
  }
}

HDF5PropertyList &HDF5PropertyList::operator=(const HDF5PropertyList &other) {
  if (id_ != H5I_INVALID_HID) {
    const herr_t err = H5Pclose(id_);
    if (err < 0) {
      NUMERIC_ERROR("Failed to close property list\n\n{}",
                    get_hdf5_stacktrace());
    }
  }
  id_ = H5Pcopy(static_cast<hid_t>(other));
  if (id_ == H5I_INVALID_HID) {
    NUMERIC_ERROR("Failed to copy property list\n\n{}", get_hdf5_stacktrace());
  }
  return *this;
}

HDF5PropertyList &HDF5PropertyList::operator=(HDF5PropertyList &&other) {
  if (id_ != H5I_INVALID_HID) {
    const herr_t err = H5Pclose(id_);
    if (err < 0) {
      NUMERIC_ERROR("Failed to close property list\n\n{}",
                    get_hdf5_stacktrace());
    }
  }
  id_ = other.id_;
  other.id_ = H5I_INVALID_HID;
  return *this;
}

HDF5PropertyList::operator hid_t() const { return id_; }

} // namespace numeric::io
