#ifndef NUMERIC_IO_FILE_MODE_HPP_
#define NUMERIC_IO_FILE_MODE_HPP_

#include <string_view>

namespace numeric::io {

/**
 * @brief Enumeration representing file modes for I/O operations.
 */
enum class FileMode {
  READ,  /**< Read mode: open existing file for reading. */
  WRITE, /**< Write mode: create a new file or overwrite existing file for
            writing. */
  KEEP_EXISTING, /**< Keep existing mode: open existing file without modifying
                    it. */
  OVERWRITE /**< Overwrite mode: overwrite existing file or create a new file
               for writing. */
};

/**
 * @brief Converts FileMode enum value to string representation.
 *
 * @param mode File mode to convert.
 * @return String representation of the file mode.
 */
std::string_view to_string(FileMode mode);

} // namespace numeric::io

#endif
