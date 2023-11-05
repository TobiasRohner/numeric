#include <numeric/utils/datatype.hpp>

namespace numeric::utils {

std::string_view to_string(Datatype type) {
  switch (type) {
  case Datatype::FLOAT:
    return "float";
  case Datatype::DOUBLE:
    return "double";
  case Datatype::LONG_DOUBLE:
    return "long double";
  case Datatype::INT8_T:
    return "int8_t";
  case Datatype::UINT8_T:
    return "uint8_t";
  case Datatype::INT16_T:
    return "int16_t";
  case Datatype::UINT16_T:
    return "uint16_t";
  case Datatype::INT32_T:
    return "int32_t";
  case Datatype::UINT32_T:
    return "uint32_t";
  case Datatype::INT64_T:
    return "int64_t";
  case Datatype::UINT64_T:
    return "uint64_t";
  default:
    return "UNKNOWN";
  }
}

} // namespace numeric::utils
