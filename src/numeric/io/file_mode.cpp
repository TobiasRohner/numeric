#include <numeric/io/file_mode.hpp>

namespace numeric::io {

std::string_view to_string(FileMode mode) {
  switch (mode) {
  case FileMode::READ:
    return "READ";
  case FileMode::WRITE:
    return "WRITE";
  case FileMode::KEEP_EXISTING:
    return "KEEP_EXISTING";
  case FileMode::OVERWRITE:
    return "OVERWRITE";
  default:
    return "UNKNOWN";
  }
}

} // namespace numeric::io
