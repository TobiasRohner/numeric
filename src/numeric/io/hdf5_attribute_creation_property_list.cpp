#include <numeric/io/hdf5_attribute_creation_property_list.hpp>
#include <numeric/io/hdf5_error.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::io {

HDF5AttributeCreationPropertyList::HDF5AttributeCreationPropertyList()
    : super(H5P_ATTRIBUTE_CREATE) {}

H5T_cset_t HDF5AttributeCreationPropertyList::get_char_encoding() const {
  H5T_cset_t encoding;
  const herr_t err = H5Pget_char_encoding(id_, &encoding);
  if (err < 0) {
    NUMERIC_ERROR("Failed to get the caracter encoding for property list\n\n{}",
                  get_hdf5_stacktrace());
  }
  return encoding;
}

void HDF5AttributeCreationPropertyList::set_char_encoding(H5T_cset_t encoding) {
  const herr_t err = H5Pset_char_encoding(id_, encoding);
  if (err < 0) {
    NUMERIC_ERROR("Failed to set the caracter encoding for property list\n\n{}",
                  get_hdf5_stacktrace());
  }
}

} // namespace numeric::io
