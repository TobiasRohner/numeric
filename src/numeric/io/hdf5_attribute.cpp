#include <numeric/io/hdf5_attribute.hpp>
#include <numeric/io/hdf5_attribute_creation_property_list.hpp>
#include <numeric/io/hdf5_dataspace.hpp>
#include <numeric/io/hdf5_error.hpp>
#include <numeric/io/hdf5_type.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::io {

HDF5Attribute::HDF5Attribute(HDF5Attribute &&other) : id_(other.id_) {
  other.id_ = H5I_INVALID_HID;
}

HDF5Attribute::~HDF5Attribute() {
  if (id_ != H5I_INVALID_HID) {
    herr_t err = H5Aclose(id_);
    if (err < 0) {
      NUMERIC_ERROR("Failed to close attribute\n\n{}", get_hdf5_stacktrace());
    }
  }
}

HDF5Attribute &HDF5Attribute::operator=(HDF5Attribute &&other) {
  if (id_ != H5I_INVALID_HID) {
    herr_t err = H5Aclose(id_);
    if (err < 0) {
      NUMERIC_ERROR("Failed to close attribute\n\n{}", get_hdf5_stacktrace());
    }
  }
  id_ = other.id_;
  other.id_ = H5I_INVALID_HID;
  return *this;
}

HDF5Attribute HDF5Attribute::open(hid_t id, std::string_view name) {
  hid_t attrid = H5Aopen(id, name.data(), H5P_DEFAULT);
  if (attrid == H5I_INVALID_HID) {
    NUMERIC_ERROR("Failed to open attribute \"{}\"\n\n{}", name,
                  get_hdf5_stacktrace());
  }
  return HDF5Attribute(attrid);
}

HDF5Attribute HDF5Attribute::create(hid_t id, std::string_view name,
                                    utils::Datatype datatype, hsize_t size) {
  HDF5Type type(datatype);
  HDF5Dataspace dataspace(1, &size);
  HDF5AttributeCreationPropertyList acpl;
  acpl.set_char_encoding(H5T_CSET_UTF8);
  const hid_t attrid = H5Acreate(id, name.data(), static_cast<hid_t>(type),
                                 static_cast<hid_t>(dataspace),
                                 static_cast<hid_t>(acpl), H5P_DEFAULT);
  if (attrid == H5I_INVALID_HID) {
    NUMERIC_ERROR("Failed to create attribute \"{}\"\n\n{}", name,
                  get_hdf5_stacktrace());
  }
  return HDF5Attribute(attrid);
}

HDF5Attribute HDF5Attribute::create_string(hid_t id, std::string_view name,
                                           hsize_t size) {
  HDF5Type type(H5Tcopy(H5T_C_S1));
  if (static_cast<hid_t>(type) == H5I_INVALID_HID) {
    NUMERIC_ERROR("Failed to create a new string datatype\n\n{}",
                  get_hdf5_stacktrace());
  }
  const herr_t err = H5Tset_size(static_cast<hid_t>(type), size);
  if (err < 0) {
    NUMERIC_ERROR("Failed to set string size\n\n{}", get_hdf5_stacktrace());
  }
  HDF5Dataspace dataspace;
  const hid_t attrid =
      H5Acreate(id, name.data(), static_cast<hid_t>(type),
                static_cast<hid_t>(dataspace), H5P_DEFAULT, H5P_DEFAULT);
  if (attrid == H5I_INVALID_HID) {
    NUMERIC_ERROR("Failed to create attribute \"{}\"\n\n{}", name,
                  get_hdf5_stacktrace());
  }
  return HDF5Attribute(attrid);
}

utils::Datatype HDF5Attribute::datatype() const {
  const hid_t id = get_type();
  return static_cast<utils::Datatype>(HDF5Type(id));
}

size_t HDF5Attribute::size() const {
  const hid_t id = H5Aget_space(id_);
  if (id == H5I_INVALID_HID) {
    NUMERIC_ERROR("Failed to get the dataspace\n\n{}", get_hdf5_stacktrace());
  }
  const HDF5Dataspace dataspace(id);
  return dataspace.dims()[0];
}

void HDF5Attribute::read(utils::Datatype dtype, void *data) const {
  const HDF5Type type(dtype);
  const herr_t err = H5Aread(id_, static_cast<hid_t>(type), data);
  if (err < 0) {
    NUMERIC_ERROR("Failed to read attribute\n\n{}", get_hdf5_stacktrace());
  }
}

void HDF5Attribute::write(utils::Datatype dtype, const void *data) {
  const HDF5Type type(dtype);
  const herr_t err = H5Awrite(id_, static_cast<hid_t>(type), data);
  if (err < 0) {
    NUMERIC_ERROR("Failed to write attribute\n\n{}", get_hdf5_stacktrace());
  }
}

void HDF5Attribute::write(std::string_view data) {
  const hid_t type = get_type();
  const herr_t err = H5Awrite(id_, type, data.data());
  if (err < 0) {
    NUMERIC_ERROR("Failed to write string attribute\n\n{}",
                  get_hdf5_stacktrace());
  }
}

HDF5Attribute::HDF5Attribute() : id_(H5I_INVALID_HID) {}

HDF5Attribute::HDF5Attribute(hid_t id) : id_(id) {}

hid_t HDF5Attribute::get_type() const {
  const hid_t id = H5Aget_type(id_);
  if (id == H5I_INVALID_HID) {
    NUMERIC_ERROR("Failed to get the datatype\n\n{}", get_hdf5_stacktrace());
  }
  return id;
}

} // namespace numeric::io
