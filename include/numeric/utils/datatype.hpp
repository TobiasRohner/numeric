#ifndef NUMERIC_UTILS_DATATYPE_HPP_
#define NUMERIC_UTILS_DATATYPE_HPP_

namespace numeric::utils {

class Datatype {
public:
  Datatype() = default;
  Datatype(const Datatype &) = default;
  Datatype(Datatype &&) = default;
  virtual ~Datatype() = default;

  Datatype &operator=(const Datatype &) = default;
  Datatype &operator=(Datatype &&) = default;
};

class DatatypeInt8 : public Datatype {
public:
  DatatypeInt8() = default;
  DatatypeInt8(const DatatypeInt8 &) = default;
  DatatypeInt8(DatatypeInt8 &&) = default;
  virtual ~DatatypeInt8() override = default;

  DatatypeInt8 &operator=(const DatatypeInt8 &) = default;
  DatatypeInt8 &operator=(DatatypeInt8 &&) = default;
};

class DatatypeUInt8 : public Datatype {
public:
  DatatypeUInt8() = default;
  DatatypeUInt8(const DatatypeUInt8 &) = default;
  DatatypeUInt8(DatatypeUInt8 &&) = default;
  virtual ~DatatypeUInt8() override = default;

  DatatypeUInt8 &operator=(const DatatypeUInt8 &) = default;
  DatatypeUInt8 &operator=(DatatypeUInt8 &&) = default;
};

} // namespace numeric::utils

#endif
