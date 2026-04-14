#ifndef NUMERIC_IO_HDF5_TYPE_HPP_
#define NUMERIC_IO_HDF5_TYPE_HPP_

#include <hdf5.h>
#include <memory>
#include <numeric/utils/datatype.hpp>
#include <vector>

namespace numeric::io {

class HDF5Type {
public:
  explicit HDF5Type(hid_t id);
  explicit HDF5Type(utils::Datatype type);
  HDF5Type(const HDF5Type &) = default;
  HDF5Type(HDF5Type &&) = default;
  HDF5Type &operator=(const HDF5Type &) = default;
  HDF5Type &operator=(HDF5Type &&) = default;
  explicit operator utils::Datatype() const;
  explicit operator hid_t() const;

  bool operator==(const HDF5Type &other) const;
  bool operator!=(const HDF5Type &other) const;

  HDF5Type native_type() const;
  HDF5Type super() const;

  bool is_integer() const;
  bool is_float() const;
  bool is_string() const;
  bool is_bitfield() const;
  bool is_opaque() const;
  bool is_compound() const;
  bool is_enum() const;
  bool is_array() const;

  std::vector<hsize_t> get_array_dims() const;

  static hid_t datatype_to_hdf5_type(utils::Datatype type);

private:
  std::shared_ptr<hid_t> id_;

  H5T_class_t get_class() const;
  int get_array_ndims() const;
};

template <typename T> HDF5Type to_hdf5_type() {
  constexpr utils::Datatype dtype = utils::to_datatype_v<T>;
  const hid_t id = HDF5Type::datatype_to_hdf5_type(dtype);
  return HDF5Type(id);
}

} // namespace numeric::io

#endif
