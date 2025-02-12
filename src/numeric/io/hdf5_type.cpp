#include <numeric/io/hdf5_error.hpp>
#include <numeric/io/hdf5_type.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::io {

HDF5Type::HDF5Type(hid_t id)
    : id_(std::shared_ptr<hid_t>(new hid_t(id),
                                 [](hid_t *id) { H5Tclose(*id); })) {}

HDF5Type::HDF5Type(utils::Datatype type)
    : HDF5Type(datatype_to_hdf5_type(type)) {}

HDF5Type::operator utils::Datatype() const {
  const HDF5Type native = native_type();
  if (native == HDF5Type(utils::Datatype::FLOAT)) {
    return utils::Datatype::FLOAT;
  }
  if (native == HDF5Type(utils::Datatype::DOUBLE)) {
    return utils::Datatype::DOUBLE;
  }
  if (native == HDF5Type(utils::Datatype::INT8_T)) {
    return utils::Datatype::INT8_T;
  }
  if (native == HDF5Type(utils::Datatype::UINT8_T)) {
    return utils::Datatype::UINT8_T;
  }
  if (native == HDF5Type(utils::Datatype::INT16_T)) {
    return utils::Datatype::INT16_T;
  }
  if (native == HDF5Type(utils::Datatype::UINT16_T)) {
    return utils::Datatype::UINT16_T;
  }
  if (native == HDF5Type(utils::Datatype::INT32_T)) {
    return utils::Datatype::INT32_T;
  }
  if (native == HDF5Type(utils::Datatype::UINT32_T)) {
    return utils::Datatype::UINT32_T;
  }
  if (native == HDF5Type(utils::Datatype::INT64_T)) {
    return utils::Datatype::INT64_T;
  }
  if (native == HDF5Type(utils::Datatype::UINT64_T)) {
    return utils::Datatype::UINT64_T;
  }
  NUMERIC_ERROR("Unsupported datatype");
}

HDF5Type::operator hid_t() const { return *id_; }

bool HDF5Type::operator==(const HDF5Type &other) const {
  const htri_t value = H5Tequal(*id_, *other.id_);
  if (value < 0) {
    NUMERIC_ERROR("Failed to compare HDF5 types\n\n{}", get_hdf5_stacktrace());
  }
  return value > 0;
}

bool HDF5Type::operator!=(const HDF5Type &other) const {
  return !(*this == other);
}

HDF5Type HDF5Type::native_type() const {
  const hid_t id = H5Tget_native_type(*id_, H5T_DIR_DEFAULT);
  if (id == H5I_INVALID_HID) {
    NUMERIC_ERROR("Unable to get native type\n\n{}", get_hdf5_stacktrace());
  }
  return HDF5Type(id);
}

HDF5Type HDF5Type::super() const {
  const hid_t id = H5Tget_super(*id_);
  if (id == H5I_INVALID_HID) {
    NUMERIC_ERROR("Unable to get super type\n\n{}", get_hdf5_stacktrace());
  }
  return HDF5Type(id);
}

bool HDF5Type::is_integer() const {
  const H5T_class_t cls = get_class();
  return cls == H5T_INTEGER;
}

bool HDF5Type::is_float() const {
  const H5T_class_t cls = get_class();
  return cls == H5T_FLOAT;
}

bool HDF5Type::is_string() const {
  const H5T_class_t cls = get_class();
  return cls == H5T_STRING;
}

bool HDF5Type::is_bitfield() const {
  const H5T_class_t cls = get_class();
  return cls == H5T_BITFIELD;
}

bool HDF5Type::is_opaque() const {
  const H5T_class_t cls = get_class();
  return cls == H5T_OPAQUE;
}

bool HDF5Type::is_compound() const {
  const H5T_class_t cls = get_class();
  return cls == H5T_COMPOUND;
}

bool HDF5Type::is_enum() const {
  const H5T_class_t cls = get_class();
  return cls == H5T_ENUM;
}

bool HDF5Type::is_array() const {
  const H5T_class_t cls = get_class();
  return cls == H5T_ARRAY;
}

std::vector<hsize_t> HDF5Type::get_array_dims() const {
  const int ndims = get_array_ndims();
  std::vector<hsize_t> dims(ndims);
  const int status = H5Tget_array_dims(*id_, dims.data());
  if (status < 0) {
    NUMERIC_ERROR("Unable to get array dimensions\n\n{}",
                  get_hdf5_stacktrace());
  }
  return dims;
}

H5T_class_t HDF5Type::get_class() const {
  const H5T_class_t cls = H5Tget_class(*id_);
  if (cls == H5T_NO_CLASS) {
    NUMERIC_ERROR("Failed to get class of HDF5 type\n\n{}",
                  get_hdf5_stacktrace());
  }
  return cls;
}

int HDF5Type::get_array_ndims() const {
  const int ndims = H5Tget_array_ndims(*id_);
  if (ndims < 0) {
    NUMERIC_ERROR("Unable to get array dimensions\n\n", get_hdf5_stacktrace());
  }
  return ndims;
}

hid_t HDF5Type::datatype_to_hdf5_type(utils::Datatype type) {
  switch (type) {
  case utils::Datatype::FLOAT:
    return H5T_NATIVE_FLOAT;
  case utils::Datatype::DOUBLE:
    return H5T_NATIVE_DOUBLE;
  case utils::Datatype::INT8_T:
    return H5T_NATIVE_INT8;
  case utils::Datatype::UINT8_T:
    return H5T_NATIVE_UINT8;
  case utils::Datatype::INT16_T:
    return H5T_NATIVE_INT16;
  case utils::Datatype::UINT16_T:
    return H5T_NATIVE_UINT16;
  case utils::Datatype::INT32_T:
    return H5T_NATIVE_INT32;
  case utils::Datatype::UINT32_T:
    return H5T_NATIVE_UINT32;
  case utils::Datatype::INT64_T:
    return H5T_NATIVE_INT64;
  case utils::Datatype::UINT64_T:
    return H5T_NATIVE_UINT64;
  default:
    NUMERIC_ERROR("Unsupported Datatype");
  }
}

template <> HDF5Type to_hdf5_type<float>() {
  return HDF5Type(H5T_NATIVE_FLOAT);
}
template <> HDF5Type to_hdf5_type<double>() {
  return HDF5Type(H5T_NATIVE_DOUBLE);
}
template <> HDF5Type to_hdf5_type<int8_t>() {
  return HDF5Type(H5T_NATIVE_INT8);
}
template <> HDF5Type to_hdf5_type<uint8_t>() {
  return HDF5Type(H5T_NATIVE_UINT8);
}
template <> HDF5Type to_hdf5_type<int16_t>() {
  return HDF5Type(H5T_NATIVE_INT16);
}
template <> HDF5Type to_hdf5_type<uint16_t>() {
  return HDF5Type(H5T_NATIVE_UINT16);
}
template <> HDF5Type to_hdf5_type<int32_t>() {
  return HDF5Type(H5T_NATIVE_INT32);
}
template <> HDF5Type to_hdf5_type<uint32_t>() {
  return HDF5Type(H5T_NATIVE_UINT32);
}
template <> HDF5Type to_hdf5_type<int64_t>() {
  return HDF5Type(H5T_NATIVE_INT64);
}
template <> HDF5Type to_hdf5_type<uint64_t>() {
  return HDF5Type(H5T_NATIVE_UINT64);
}

} // namespace numeric::io
