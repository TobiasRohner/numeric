#ifndef NUMERIC_IO_NETCDF_ERROR_HPP_
#define NUMERIC_IO_NETCDF_ERROR_HPP_

#include <netcdf.h>
#include <numeric/utils/error.hpp>

namespace numeric::io {

#define NUMERIC_CHECK_NETCDF_STATUS(...)                                       \
  if (int status = (__VA_ARGS__); status != NC_NOERR) {                        \
    NUMERIC_ERROR("NetCDF error: {}", nc_strerror(status));                    \
  }

} // namespace numeric::io

#endif
