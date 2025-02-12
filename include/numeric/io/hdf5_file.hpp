#ifndef NUMERIC_IO_HDF5_FILE_HPP_
#define NUMERIC_IO_HDF5_FILE_HPP_

#include <memory>
#include <numeric/io/file_mode.hpp>
#include <numeric/io/hdf5_group.hpp>
#include <string_view>

namespace numeric::io {

/**
 * @brief A HDF5 file
 *
 * This class provides an interface to open, create,
 * and modify HDF5 files.
 */
class HDF5File : public HDF5Group {
  using super = HDF5Group;

public:
  HDF5File(const HDF5File &) = delete;
  HDF5File(HDF5File &&) = delete;
  HDF5File &operator=(const HDF5File &) = delete;
  HDF5File &operator=(HDF5File &&) = delete;
  virtual ~HDF5File() override = default;

  /**
   * @brief Open an existing HDF5 dataset
   *
   * Opens an existing HDF5 dataset located at the given path.
   * Default mode is opening with read only access.
   *
   * @param path The path to the HDF5 dataset
   * @param mode File access properties (see READ, WRITE)
   * @returns An std::shared_ptr to a HDF5File
   */
  static std::shared_ptr<HDF5File> open(const std::string_view &path,
                                        FileMode mode = FileMode::READ);
  /**
   * @brief Create a new HDF5 dataset
   *
   * Creates a new HDF5 dataset located at the given path.
   * The default behavior is to override the contents of the
   * file should it already exist.
   *
   * @param path The path to the HDF5 dataset
   * @param mode File access properties (see KEEP_EXISTING, OVERWRITE)
   * @returns An std::shared_ptr to a HDF5
   */
  static std::shared_ptr<HDF5File> create(const std::string_view &path,
                                          FileMode mode = FileMode::OVERWRITE);

protected:
  HDF5File(hid_t id);
};

} // namespace numeric::io

#endif
