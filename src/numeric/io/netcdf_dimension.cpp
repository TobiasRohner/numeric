#include <numeric/io/netcdf_dimension.hpp>
#include <numeric/io/netcdf_error.hpp>

namespace numeric::io {

size_t NetCDFDimension::get_size() const {
  size_t len;
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_dimlen(ncid_, dimid_, &len));
  return len;
}

std::string NetCDFDimension::get_name() const {
  char name[NC_MAX_NAME];
  NUMERIC_CHECK_NETCDF_STATUS(nc_inq_dimname(ncid_, dimid_, name));
  return name;
}

NetCDFDimension::NetCDFDimension(
    int ncid, int dimid, const std::shared_ptr<const NetCDFFile> &root_file)
    : root_file_(root_file), ncid_(ncid), dimid_(dimid) {}

} // namespace numeric::io
