#include <cctype>
#include <fstream>
#include <iostream>
#include <numeric/io/gmsh_reader.hpp>
#include <numeric/utils/endian.hpp>

namespace numeric::io {

GmshReaderBase::FileType GmshReaderBase::file_type() const noexcept {
  return file_type_;
}

template <typename T>
T GmshReaderBase::read_value_ascii(std::istream &is) const {
  T value;
  is >> value;
  return value;
}

template <typename T>
T GmshReaderBase::read_value_binary(std::istream &is) const {
  T value;
  if (endianness_ == std::endian::native) {
    is.read(reinterpret_cast<char *>(&value), sizeof(T));
  } else {
    static constexpr std::endian nonnative =
        std::endian::native == std::endian::little ? std::endian::big
                                                   : std::endian::little;
    utils::Endian<T, nonnative> tmp;
    is.read(reinterpret_cast<char *>(&tmp), sizeof(T));
    value = tmp;
  }
  return value;
}

template <typename T> T GmshReaderBase::read_value(std::istream &is) const {
  if (file_type() == ASCII) {
    return read_value_ascii<T>(is);
  } else {
    return read_value_binary<T>(is);
  }
}

template <typename T>
std::vector<T> GmshReaderBase::read_values_ascii(std::istream &is,
                                                 size_t N) const {
  std::vector<T> values(N);
  for (size_t i = 0; i < N; ++i) {
    is >> values[i];
  }
  return values;
}

template <typename T>
std::vector<T> GmshReaderBase::read_values_binary(std::istream &is,
                                                  size_t N) const {
  std::vector<T> values(N);
  if (endianness_ == std::endian::native) {
    is.read(reinterpret_cast<char *>(values.data()), N * sizeof(T));
  } else {
    static constexpr std::endian nonnative =
        std::endian::native == std::endian::little ? std::endian::big
                                                   : std::endian::little;
    std::vector<utils::Endian<T, nonnative>> tmp(N);
    is.read(reinterpret_cast<char *>(tmp.data()), N * sizeof(T));
    for (size_t i = 0; i < N; ++i) {
      values[i] = tmp[i];
    }
  }
  return values;
}

template <typename T>
std::vector<T> GmshReaderBase::read_values(std::istream &is, size_t N) const {
  if (file_type() == ASCII) {
    return read_values_ascii<T>(is, N);
  } else {
    return read_values_binary<T>(is, N);
  }
}

void GmshReaderBase::consume_whitespace(std::istream &is) {
  int ch;
  do {
    ch = is.get();
  } while (std::isspace(ch));
  is.unget();
}

bool GmshReaderBase::parse_section(std::istream &is) {
  const auto fail = [&is, pos = is.tellg()]() {
    is.seekg(pos);
    return false;
  };
  consume_whitespace(is);
  if (is.get() != '$') {
    if (is.good()) {
      return fail();
    } else {
      return true;
    }
  }
  const std::string section_name = [&]() {
    std::string name;
    is >> name;
    is.get();
    return name;
  }();
  bool res;
  if (section_name == "MeshFormat") {
    res = parse_mesh_format(is);
  } else if (section_name == "Nodes") {
    res = parse_nodes(is);
  } else if (section_name == "Elements") {
    res = parse_elements(is);
  } else {
    res = parse_unknown(is, section_name);
  }
  if (!res) {
    return fail();
  }
  return true;
}

bool GmshReaderBase::parse_mesh_format(std::istream &is) {
  is >> version_;
  int ft;
  is >> ft;
  file_type_ = ft == 0 ? ASCII : BINARY;
  is >> data_size_;
  if (file_type() == BINARY) {
    char one[sizeof(int)];
    is.get();
    is.read(one, sizeof(int));
    if (one[0] == 1) {
      endianness_ = std::endian::little;
    } else {
      endianness_ = std::endian::big;
    }
  }
  consume_whitespace(is);
  std::string end_tag;
  is >> end_tag;
  return end_tag == "$EndMeshFormat";
}

bool GmshReaderBase::parse_unknown(std::istream &is,
                                   const std::string &name) const {
  std::cerr << "GmshReaderBase: Unknown section \"" << name << "\"\n";
  std::string closing = "$End" + name;
  int matches_until = 0;
  while (matches_until < closing.size()) {
    if (is.eof()) {
      return false;
    }
    if (is.get() == closing[matches_until]) {
      ++matches_until;
    } else {
      matches_until = 0;
    }
  }
  return true;
}

size_t GmshReaderBase::node_tag_to_idx(size_t tag, size_t min_node_tag) {
  if (nodes_are_contiguous_) {
    return tag - min_node_tag;
  } else {
    auto tag_to_idx = node_tag_to_idx_map_.find(tag);
    if (tag_to_idx == node_tag_to_idx_map_.end()) {
      const size_t idx = node_tag_to_idx_map_.size();
      tag_to_idx->second = idx;
      return idx;
    } else {
      return tag_to_idx->second;
    }
  }
}

template size_t
GmshReaderBase::read_value_ascii<size_t>(std::istream &is) const;
template size_t
GmshReaderBase::read_value_binary<size_t>(std::istream &is) const;
template size_t GmshReaderBase::read_value<size_t>(std::istream &is) const;
template std::vector<size_t>
GmshReaderBase::read_values_ascii<size_t>(std::istream &is, size_t N) const;
template std::vector<size_t>
GmshReaderBase::read_values_binary<size_t>(std::istream &is, size_t N) const;
template std::vector<size_t>
GmshReaderBase::read_values<size_t>(std::istream &is, size_t N) const;
template int GmshReaderBase::read_value_ascii<int>(std::istream &is) const;
template int GmshReaderBase::read_value_binary<int>(std::istream &is) const;
template int GmshReaderBase::read_value<int>(std::istream &is) const;
template std::vector<int>
GmshReaderBase::read_values_ascii<int>(std::istream &is, size_t N) const;
template std::vector<int>
GmshReaderBase::read_values_binary<int>(std::istream &is, size_t N) const;
template std::vector<int> GmshReaderBase::read_values<int>(std::istream &is,
                                                           size_t N) const;
template double
GmshReaderBase::read_value_ascii<double>(std::istream &is) const;
template double
GmshReaderBase::read_value_binary<double>(std::istream &is) const;
template double GmshReaderBase::read_value<double>(std::istream &is) const;
template std::vector<double>
GmshReaderBase::read_values_ascii<double>(std::istream &is, size_t N) const;
template std::vector<double>
GmshReaderBase::read_values_binary<double>(std::istream &is, size_t N) const;
template std::vector<double>
GmshReaderBase::read_values<double>(std::istream &is, size_t N) const;

} // namespace numeric::io
