#include <numeric/io/hdf5_error.hpp>
#include <numeric/io/hdf5_file.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::io {

static unsigned file_mode_to_hdf5_mode(FileMode mode) {
  switch (mode) {
  case FileMode::READ:
    return H5F_ACC_RDONLY;
  case FileMode::WRITE:
    return H5F_ACC_RDWR;
  case FileMode::KEEP_EXISTING:
    return H5F_ACC_EXCL;
  case FileMode::OVERWRITE:
    return H5F_ACC_TRUNC;
  default:
    NUMERIC_ERROR("Unsupported file mode {}\n\n{}", to_string(mode),
                  get_hdf5_stacktrace());
    return 0;
  }
}

std::shared_ptr<HDF5File> HDF5File::open(const std::string_view &path,
                                         FileMode mode) {
  const hid_t id =
      H5Fopen(path.data(), file_mode_to_hdf5_mode(mode), H5P_DEFAULT);
  if (id == H5I_INVALID_HID) {
    NUMERIC_ERROR("Error opening file\n\n{}", get_hdf5_stacktrace());
  }
  return std::shared_ptr<HDF5File>(new HDF5File(id));
}

std::shared_ptr<HDF5File> HDF5File::create(const std::string_view &path,
                                           FileMode mode) {
  const hid_t id = H5Fcreate(path.data(), file_mode_to_hdf5_mode(mode),
                             H5P_DEFAULT, H5P_DEFAULT);
  if (id == H5I_INVALID_HID) {
    NUMERIC_ERROR("Error creating file\n\n{}", get_hdf5_stacktrace());
  }
  return std::shared_ptr<HDF5File>(new HDF5File(id));
}

HDF5File::HDF5File(hid_t id) : super(id, nullptr) {}

} // namespace numeric::io
