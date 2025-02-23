#ifndef NUMERIC_IO_ATTRIBUTABLE_HPP_
#define NUMERIC_IO_ATTRIBUTABLE_HPP_

#include <numeric/utils/datatype.hpp>
#include <numeric/utils/error.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace numeric::io {

/**
 * @brief Implements writing of attributes to child classes
 *
 * Some data formats enable the writing of attributes to
 * store some metadata about the rest of the data stored in
 * the file (see HDF5Group, HDF5Variable, NetCDFFile, NetCDFVariable).
 * This class provides a unified interface to read and
 * write attributes.
 */
class Attributable {
public:
  virtual ~Attributable() = default;

  /**
   * @brief Get the names of all attributes
   *
   * Returns the names of all attributes associated with this
   * Attributable as a vector of strings.
   *
   * @returns A vector of all attribute names
   */
  std::vector<std::string> get_attribute_names() const {
    return do_get_attribute_names();
  }

  bool attribute_exists(std::string_view name) const {
    return do_attribute_exists(name);
  }

  /**
   * @brief Get the utils::Datatype of an attribute
   *
   * @param name The name of the attribute
   * @returns The utils::Datatype of the attribute with name `name`
   */
  utils::Datatype get_attribute_datatype(std::string_view name) const {
    return do_get_attribute_datatype(name);
  }

  /**
   * @brief Get the number of elements stored in an attribute
   *
   * @param name The name of the attribute
   * @returns The number of elements contained in the attribute with name `name`
   */
  size_t get_attribute_length(std::string_view name) const {
    return do_get_attribute_length(name);
  }

  /**
   * @brief Read an attribute of given type
   *
   * Reads the attribute with name `name` into a vector of objects
   * of type `T`. The type given to the function in the template
   * argument must match the type of the attribute. Otherwise a
   * std::invalid_argument exception is thrown.
   *
   * @tparam T The type of the attribute
   * @param name The name of the attribute
   * @returns A vector containing the attribute data
   */
  template <typename T>
  std::vector<T> read_attribute(std::string_view name) const {
    std::vector<T> attr;
    const size_t attrlen = get_attribute_length(name);
    attr.resize(attrlen);
    do_read_attribute(name, utils::to_datatype_v<T>, attr.data());
    return attr;
  }

  /**
   * @brief Write a new attribute
   *
   * @tparam T The type of the attribute
   * @param name The name of the attribute
   * @param attr The data to write into the attribute
   */
  template <typename T>
  void write_attribute(std::string_view name, const std::vector<T> &attr) {
    do_write_attribute(name, attr.data(), attr.size(), utils::to_datatype_v<T>);
  }

  /**
   * @brief Write a new string attribute
   *
   * @param name The name of the attribute
   * @param attr The string to write into the attribute
   */
  void write_attribute(std::string_view name, std::string_view attr);

protected:
  virtual std::vector<std::string> do_get_attribute_names() const = 0;
  virtual bool do_attribute_exists(std::string_view name) const = 0;
  virtual utils::Datatype
  do_get_attribute_datatype(std::string_view name) const = 0;
  virtual size_t do_get_attribute_length(std::string_view name) const = 0;
  virtual void do_read_attribute(std::string_view name,
                                 utils::Datatype datatype,
                                 void *attr) const = 0;
  virtual void do_write_attribute(std::string_view name, const void *attr,
                                  size_t len, utils::Datatype datatype) = 0;
  virtual void do_write_attribute(std::string_view name,
                                  std::string_view attr) = 0;
};

} // namespace numeric::io

#endif
