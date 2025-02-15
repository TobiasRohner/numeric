#ifndef NUMERIC_IO_HDF5_ATTRIBUTE_CREATION_PROPERTY_LIST_HPP_
#define NUMERIC_IO_HDF5_ATTRIBUTE_CREATION_PROPERTY_LIST_HPP_

#include <numeric/io/hdf5_property_list.hpp>

namespace numeric::io {

class HDF5AttributeCreationPropertyList : public HDF5PropertyList {
  using super = HDF5PropertyList;

public:
  HDF5AttributeCreationPropertyList();
  HDF5AttributeCreationPropertyList(const HDF5AttributeCreationPropertyList &) =
      default;
  HDF5AttributeCreationPropertyList(HDF5AttributeCreationPropertyList &&) =
      default;
  ~HDF5AttributeCreationPropertyList() = default;
  HDF5AttributeCreationPropertyList &
  operator=(const HDF5AttributeCreationPropertyList &) = default;
  HDF5AttributeCreationPropertyList &
  operator=(HDF5AttributeCreationPropertyList &&) = default;

  using super::operator hid_t;

  H5T_cset_t get_char_encoding() const;
  void set_char_encoding(H5T_cset_t encoding);
};

} // namespace numeric::io

#endif
