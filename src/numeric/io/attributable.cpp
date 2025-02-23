#include <numeric/io/attributable.hpp>

namespace numeric::io {

void Attributable::write_attribute(std::string_view name,
                                   std::string_view attr) {
  do_write_attribute(name, attr);
}

} // namespace numeric::io
