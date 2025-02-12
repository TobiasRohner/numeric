#ifndef NUMERIC_IO_HDF5_DATASPACE_HPP_
#define NUMERIC_IO_HDF5_DATASPACE_HPP_

#include <hdf5.h>
#include <numeric/config.hpp>
#include <vector>

namespace numeric::io {

class HDF5Dataspace {
public:
  HDF5Dataspace();
  explicit HDF5Dataspace(hid_t id);
  HDF5Dataspace(int ndims, const hsize_t *dims);
  HDF5Dataspace(const HDF5Dataspace &) = delete;
  HDF5Dataspace(HDF5Dataspace &&);
  ~HDF5Dataspace();
  HDF5Dataspace &operator=(const HDF5Dataspace &) = delete;
  HDF5Dataspace &operator=(HDF5Dataspace &&);
  explicit operator hid_t() const;

  std::vector<dim_t> dims() const;
  void select_hyperslab(const hsize_t *start, const hsize_t *stride,
                        const hsize_t *count);

private:
  hid_t id_;

  size_t num_dims() const;
};

} // namespace numeric::io

#endif
