#include <fmt/core.h>
#include <hdf5.h>
#include <numeric/io/hdf5_error.hpp>

namespace numeric::io {

extern "C" {
static herr_t append_to_string(unsigned n, const H5E_error_t *err_desc,
                               void *client_data) {
  static constexpr int MSG_SIZE = 128;
  std::string *msg = static_cast<std::string *>(client_data);
  char maj[MSG_SIZE];
  char min[MSG_SIZE];
  char cls[MSG_SIZE];
  H5E_type_t type;
  H5Eget_class_name(err_desc->cls_id, cls, MSG_SIZE);
  H5Eget_msg(err_desc->maj_num, &type, maj, MSG_SIZE);
  H5Eget_msg(err_desc->min_num, &type, min, MSG_SIZE);
  const bool has_desc =
      err_desc->desc != NULL && std::strlen(err_desc->desc) != 0;
  *msg += fmt::format("#{:03}: {}:{} in {}{}{}\n", n,
                      err_desc->file_name ? err_desc->file_name : "",
                      err_desc->line,
                      err_desc->func_name ? err_desc->func_name : "",
                      has_desc ? ": " : "", has_desc ? err_desc->desc : "");
  *msg += fmt::format("  major: {}\n", maj);
  *msg += fmt::format("  minor: {}\n", min);
  return 0;
}
}

std::string get_hdf5_stacktrace() {
  std::string msg;
  H5Ewalk(H5E_DEFAULT, H5E_WALK_DOWNWARD, append_to_string, &msg);
  return msg;
}

} // namespace numeric::io
