#ifndef NUMERIC_IO_NETCDF_VARIABLE_HPP_
#define NUMERIC_IO_NETCDF_VARIABLE_HPP_

#include <memory>
#include <netcdf.h>
#include <numeric/io/variable.hpp>

namespace numeric::io {

class NetCDFFile;

class NetCDFVariable : public Variable {
  using super = Variable;

public:
  NetCDFVariable(int ncid, int varid,
                 const std::shared_ptr<const NetCDFFile> &root_file);
  virtual ~NetCDFVariable() override = default;

  using super::datatype;
  using super::dims;
  using super::read;
  using super::write;

protected:
  int ncid_;
  int varid_;
  std::shared_ptr<const NetCDFFile> root_file_;

  virtual utils::Datatype do_datatype() const override;
  virtual std::vector<dim_t> do_dims() const override;
  virtual void do_read(void *data, const dim_t *shape, const dim_t *stride,
                       dim_t N) const override;
  virtual void do_write(const void *data, const dim_t *shape,
                        const dim_t *stride, dim_t N) override;

private:
  int get_num_dims() const;
  std::vector<int> get_dimids() const;
  dim_t get_dim(int dimid) const;
};

} // namespace numeric::io

#endif
