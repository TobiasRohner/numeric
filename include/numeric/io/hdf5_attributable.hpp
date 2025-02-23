#ifndef NUMERIC_IO_HDF5_ATTRIBUTABLE_HPP_
#define NUMERIC_IO_HDF5_ATTRIBUTABLE_HPP_

#include <hdf5.h>
#include <numeric/io/attributable.hpp>

namespace numeric::io {

class HDF5Attributable : public Attributable {
  using super = Attributable;

public:
  HDF5Attributable(hid_t id);
  HDF5Attributable(const HDF5Attributable &) = default;
  HDF5Attributable(HDF5Attributable &&) = default;
  virtual ~HDF5Attributable() override = default;
  HDF5Attributable &operator=(const HDF5Attributable &) = default;
  HDF5Attributable &operator=(HDF5Attributable &&) = default;

  using super::attribute_exists;
  using super::get_attribute_datatype;
  using super::get_attribute_length;
  using super::get_attribute_names;
  using super::read_attribute;
  using super::write_attribute;

protected:
  virtual std::vector<std::string> do_get_attribute_names() const override;
  virtual bool do_attribute_exists(std::string_view name) const override;
  virtual utils::Datatype
  do_get_attribute_datatype(std::string_view name) const override;
  virtual size_t do_get_attribute_length(std::string_view name) const override;
  virtual void do_read_attribute(std::string_view name,
                                 utils::Datatype datatype,
                                 void *attr) const override;
  virtual void do_write_attribute(std::string_view name, const void *attr,
                                  size_t len,
                                  utils::Datatype datatype) override;
  virtual void do_write_attribute(std::string_view name,
                                  std::string_view attr) override;

private:
  hid_t id_;
};

} // namespace numeric::io

#endif
