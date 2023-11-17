#ifndef NUMERIC_IO_NETCDF_DIMENSION_HPP_
#define NUMERIC_IO_NETCDF_DIMENSION_HPP_

#include <memory>
#include <string>

namespace numeric::io {

class NetCDFFile;

class NetCDFDimension {
  friend NetCDFFile;

public:
  NetCDFDimension(const NetCDFDimension &) = default;
  NetCDFDimension(NetCDFDimension &&) = default;
  NetCDFDimension &operator=(const NetCDFDimension &) = default;
  NetCDFDimension &operator=(NetCDFDimension &&) = default;
  ~NetCDFDimension() = default;

  size_t get_size() const;
  std::string get_name() const;

private:
  std::shared_ptr<const NetCDFFile> root_file_;
  int ncid_;
  int dimid_;

  NetCDFDimension(int ncid, int dimid,
                  const std::shared_ptr<const NetCDFFile> &root_file);
};

} // namespace numeric::io

#endif
