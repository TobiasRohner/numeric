#ifndef NUMERIC_IO_HDF5_ATTRIBUTE_HPP_
#define NUMERIC_IO_HDF5_ATTRIBUTE_HPP_

#include <hdf5.h>
#include <numeric/utils/datatype.hpp>
#include <string_view>

namespace numeric::io {

class HDF5Attribute {
public:
  HDF5Attribute(const HDF5Attribute &) = delete;
  HDF5Attribute(HDF5Attribute &&);
  ~HDF5Attribute();
  HDF5Attribute &operator=(const HDF5Attribute &) = delete;
  HDF5Attribute &operator=(HDF5Attribute &&);

  static HDF5Attribute open(hid_t id, std::string_view name);
  static HDF5Attribute create(hid_t id, std::string_view name,
                              utils::Datatype datatype, hsize_t size);

  utils::Datatype datatype() const;
  size_t size() const;
  void read(utils::Datatype dtype, void *data) const;
  void write(utils::Datatype dtype, const void *data);

private:
  hid_t id_;

  HDF5Attribute();
  HDF5Attribute(hid_t id);
};

} // namespace numeric::io

#endif
