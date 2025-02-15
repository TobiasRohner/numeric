#include <numeric/io/hdf5_attributable.hpp>
#include <numeric/io/hdf5_attribute.hpp>
#include <numeric/io/hdf5_error.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::io {

HDF5Attributable::HDF5Attributable(hid_t id) : id_(id) {}

extern "C" {
static herr_t push_back_attributes(hid_t id, const char *name,
                                   const H5A_info_t *, void *op_data) {
  std::vector<std::string> *attribute_names =
      static_cast<std::vector<std::string> *>(op_data);
  attribute_names->emplace_back(name);
  return 0;
}
}

std::vector<std::string> HDF5Attributable::do_get_attribute_names() const {
  std::vector<std::string> attribute_names;
  const herr_t status = H5Aiterate(id_, H5_INDEX_NAME, H5_ITER_NATIVE, NULL,
                                   push_back_attributes, &attribute_names);
  if (status < 0) {
    NUMERIC_ERROR("Failed to get attribute names\n\n{}", get_hdf5_stacktrace());
  }
  return attribute_names;
}

bool HDF5Attributable::do_attribute_exists(std::string_view name) const {
  const htri_t status = H5Aexists(id_, name.data());
  if (status < 0) {
    NUMERIC_ERROR("Failed to check if attribute exists\n\n{}",
                  get_hdf5_stacktrace());
  }
  return status;
}

utils::Datatype
HDF5Attributable::do_get_attribute_datatype(std::string_view name) const {
  const HDF5Attribute attribute = HDF5Attribute::open(id_, name);
  return attribute.datatype();
}

size_t HDF5Attributable::do_get_attribute_length(std::string_view name) const {
  const HDF5Attribute attribute = HDF5Attribute::open(id_, name);
  return attribute.size();
}

void HDF5Attributable::do_read_attribute(std::string_view name,
                                         utils::Datatype datatype,
                                         void *attr) const {
  const HDF5Attribute attribute = HDF5Attribute::open(id_, name);
  attribute.read(datatype, attr);
}

void HDF5Attributable::do_write_attribute(std::string_view name,
                                          const void *attr, size_t len,
                                          utils::Datatype datatype) {
  HDF5Attribute attribute = HDF5Attribute::create(id_, name, datatype, len);
  attribute.write(datatype, attr);
}

} // namespace numeric::io
