#ifndef NUMERIC_META_TYPE_TAG_HPP_
#define NUMERIC_META_TYPE_TAG_HPP_

namespace numeric::meta {

template <typename T> struct type_tag { using type = T; };

} // namespace numeric::meta

#endif
