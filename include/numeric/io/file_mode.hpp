#ifndef NUMERIC_IO_FILE_MODE_HPP_
#define NUMERIC_IO_FILE_MODE_HPP_

#include <string_view>

namespace numeric::io {

enum class FileMode { READ, WRITE, KEEP_EXISTING, OVERWRITE };

std::string_view to_string(FileMode mode);

} // namespace numeric::io

#endif
