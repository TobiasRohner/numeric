#include <numeric/io/netcdf_error.hpp>
#include <numeric/io/netcdf_types.hpp>
#include <numeric/io/netcdf_variable.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::io {

NetCDFVariable::NetCDFVariable(
    int ncid, int varid, const std::shared_ptr<const NetCDFFile> &root_file)
    : ncid_(ncid), varid_(varid), root_file_(root_file) {}

utils::Datatype NetCDFVariable::do_datatype() const {
  nc_type type;
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_vartype(ncid_, varid_, &type));
  return netcdf_type_to_datatype(type);
}

std::vector<dim_t> NetCDFVariable::do_dims() const {
  const std::vector<int> dimids = get_dimids();
  std::vector<dim_t> dims;
  for (int dimid : dimids) {
    dims.push_back(get_dim(dimid));
  }
  return dims;
}

void NetCDFVariable::do_read(void *data, const dim_t *shape,
                             const dim_t *stride, dim_t N) const {
  NUMERIC_ERROR("Not yet implemented");
}

void NetCDFVariable::do_write(const void *data, const dim_t *shape,
                              const dim_t *stride, dim_t N) {
  NUMERIC_ERROR("Not yet implemented");
}

int NetCDFVariable::get_num_dims() const {
  int ndims;
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_varndims(ncid_, varid_, &ndims));
  return ndims;
}

std::vector<int> NetCDFVariable::get_dimids() const {
  const int ndims = get_num_dims();
  std::vector<int> dimids(ndims);
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_vardimid(ncid_, varid_, dimids.data()));
  return dimids;
}

dim_t NetCDFVariable::get_dim(int dimid) const {
  size_t dim;
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_dimlen(ncid_, dimid, &dim));
  return dim;
}

} // namespace numeric::io
