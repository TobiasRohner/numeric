#ifndef NUMERIC_IO_NETCDF_TYPES_HPP_
#define NUMERIC_IO_NETCDF_TYPES_HPP_

#include <netcdf.h>
#include <numeric/utils/datatype.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::io {

template <typename T> struct to_netcdf_type {
  static_assert(!meta::is_same_v<T, T>, "Unsupported Datatype");
};

template <> struct to_netcdf_type<float> {
  static constexpr nc_type value = NC_FLOAT;
};
template <> struct to_netcdf_type<double> {
  static constexpr nc_type value = NC_DOUBLE;
};
template <> struct to_netcdf_type<int8_t> {
  static constexpr nc_type value = NC_BYTE;
};
template <> struct to_netcdf_type<uint8_t> {
  static constexpr nc_type value = NC_UBYTE;
};
template <> struct to_netcdf_type<int16_t> {
  static constexpr nc_type value = NC_SHORT;
};
template <> struct to_netcdf_type<uint16_t> {
  static constexpr nc_type value = NC_USHORT;
};
template <> struct to_netcdf_type<int32_t> {
  static constexpr nc_type value = NC_INT;
};
template <> struct to_netcdf_type<uint32_t> {
  static constexpr nc_type value = NC_UINT;
};
template <> struct to_netcdf_type<int64_t> {
  static constexpr nc_type value = NC_INT64;
};
template <> struct to_netcdf_type<uint64_t> {
  static constexpr nc_type value = NC_UINT64;
};

template <typename T>
static constexpr nc_type to_netcdf_type_v = to_netcdf_type<T>::value;

template <nc_type type> struct from_netcdf_type {
  static_assert(type != type, "Unsupported Datatype");
};

template <> struct from_netcdf_type<NC_FLOAT> {
  using type = float;
};
template <> struct from_netcdf_type<NC_DOUBLE> {
  using type = double;
};
template <> struct from_netcdf_type<NC_BYTE> {
  using type = int8_t;
};
template <> struct from_netcdf_type<NC_UBYTE> {
  using type = uint8_t;
};
template <> struct from_netcdf_type<NC_SHORT> {
  using type = int16_t;
};
template <> struct from_netcdf_type<NC_USHORT> {
  using type = uint16_t;
};
template <> struct from_netcdf_type<NC_INT> {
  using type = int32_t;
};
template <> struct from_netcdf_type<NC_UINT> {
  using type = uint32_t;
};
template <> struct from_netcdf_type<NC_INT64> {
  using type = int64_t;
};
template <> struct from_netcdf_type<NC_UINT64> {
  using type = uint64_t;
};

template <nc_type type>
using from_netcdf_type_t = typename from_netcdf_type<type>::type;

constexpr nc_type datatype_to_netcdf_type(utils::Datatype type) {
  switch (type) {
  case utils::Datatype::FLOAT:
    return to_netcdf_type_v<utils::from_datatype_t<utils::Datatype::FLOAT>>;
  case utils::Datatype::DOUBLE:
    return to_netcdf_type_v<utils::from_datatype_t<utils::Datatype::DOUBLE>>;
  case utils::Datatype::INT8_T:
    return to_netcdf_type_v<utils::from_datatype_t<utils::Datatype::INT8_T>>;
  case utils::Datatype::UINT8_T:
    return to_netcdf_type_v<utils::from_datatype_t<utils::Datatype::UINT8_T>>;
  case utils::Datatype::INT16_T:
    return to_netcdf_type_v<utils::from_datatype_t<utils::Datatype::INT16_T>>;
  case utils::Datatype::UINT16_T:
    return to_netcdf_type_v<utils::from_datatype_t<utils::Datatype::UINT16_T>>;
  case utils::Datatype::INT32_T:
    return to_netcdf_type_v<utils::from_datatype_t<utils::Datatype::INT32_T>>;
  case utils::Datatype::UINT32_T:
    return to_netcdf_type_v<utils::from_datatype_t<utils::Datatype::UINT32_T>>;
  case utils::Datatype::INT64_T:
    return to_netcdf_type_v<utils::from_datatype_t<utils::Datatype::INT64_T>>;
  case utils::Datatype::UINT64_T:
    return to_netcdf_type_v<utils::from_datatype_t<utils::Datatype::UINT64_T>>;
  default:
    return NC_NAT;
  }
}

constexpr utils::Datatype netcdf_type_to_datatype(nc_type type) {
  switch (type) {
  case NC_FLOAT:
    return utils::to_datatype_v<from_netcdf_type_t<NC_FLOAT>>;
  case NC_DOUBLE:
    return utils::to_datatype_v<from_netcdf_type_t<NC_DOUBLE>>;
  case NC_BYTE:
    return utils::to_datatype_v<from_netcdf_type_t<NC_BYTE>>;
  case NC_UBYTE:
    return utils::to_datatype_v<from_netcdf_type_t<NC_UBYTE>>;
  case NC_SHORT:
    return utils::to_datatype_v<from_netcdf_type_t<NC_SHORT>>;
  case NC_USHORT:
    return utils::to_datatype_v<from_netcdf_type_t<NC_USHORT>>;
  case NC_INT:
    return utils::to_datatype_v<from_netcdf_type_t<NC_INT>>;
  case NC_UINT:
    return utils::to_datatype_v<from_netcdf_type_t<NC_UINT>>;
  case NC_INT64:
    return utils::to_datatype_v<from_netcdf_type_t<NC_INT64>>;
  case NC_UINT64:
    return utils::to_datatype_v<from_netcdf_type_t<NC_UINT64>>;
  default:
    NUMERIC_ERROR("Unsupported Datatype");
  }
}

} // namespace numeric::io

#endif
