#ifndef NUMERIC_IO_HDF5_PROPERTY_LIST_HPP_
#define NUMERIC_IO_HDF5_PROPERTY_LIST_HPP_

#include <hdf5.h>

namespace numeric::io {

class HDF5PropertyList {
public:
  HDF5PropertyList();
  HDF5PropertyList(hid_t cls_id);
  HDF5PropertyList(const HDF5PropertyList &);
  HDF5PropertyList(HDF5PropertyList &&);
  ~HDF5PropertyList();
  HDF5PropertyList &operator=(const HDF5PropertyList &);
  HDF5PropertyList &operator=(HDF5PropertyList &&);

  explicit operator hid_t() const;

protected:
  hid_t id_;
};

} // namespace numeric::io

#endif
