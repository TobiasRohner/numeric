#ifndef NUMERIC_IO_HDF5_VARIABLE_HPP_
#define NUMERIC_IO_HDF5_VARIABLE_HPP_

#include <hdf5.h>
#include <numeric/io/hdf5_dataspace.hpp>
#include <numeric/io/hdf5_type.hpp>
#include <numeric/io/variable.hpp>

namespace numeric::io {

// Forward declaration
class HDF5Group;

class HDF5Variable : public Variable {
  using super = Variable;

public:
  HDF5Variable(hid_t id, const std::shared_ptr<const HDF5Group> &root_file);
  virtual ~HDF5Variable() override;

  using super::datatype;
  using super::dims;
  using super::read;
  using super::write;

protected:
  hid_t id_;
  std::shared_ptr<const HDF5Group> root_file_;

  virtual utils::Datatype do_datatype() const override;
  virtual std::vector<dim_t> do_dims() const override;
  virtual void do_read(void *data, const dim_t *shape, const dim_t *stride,
                       dim_t N) const override;
  virtual void do_write(const void *data, const dim_t *shape,
                        const dim_t *stride, dim_t N) override;

private:
  HDF5Type get_datatype() const;
  HDF5Dataspace get_dataspace() const;
};

} // namespace numeric::io

#endif
