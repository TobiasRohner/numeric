#include <gtest/gtest.h>

#include <numeric/mesh/elements.hpp>
#include <numeric/mesh/subelement_relation.hpp>
#include <numeric/mesh/unstructured_mesh.hpp>

numeric::mesh::UnstructuredMesh<double, numeric::mesh::Tria<1>,
                                numeric::mesh::Quad<1>>
generate_mesh() {
  numeric::mesh::UnstructuredMesh<double, numeric::mesh::Tria<1>,
                                  numeric::mesh::Quad<1>>
      mesh(2, 9, numeric::memory::MemoryType::HOST, 4, 2);
  auto vertices = mesh.vertices();
  vertices(0, 0) = -1;
  vertices(0, 1) = 0;
  vertices(0, 2) = 1;
  vertices(0, 3) = -1;
  vertices(0, 4) = 0;
  vertices(0, 5) = 1;
  vertices(0, 6) = -1;
  vertices(0, 7) = 0;
  vertices(0, 8) = 1;
  vertices(1, 0) = -1;
  vertices(1, 1) = -1;
  vertices(1, 2) = -1;
  vertices(1, 3) = 0;
  vertices(1, 4) = 0;
  vertices(1, 5) = 0;
  vertices(1, 6) = 1;
  vertices(1, 7) = 1;
  vertices(1, 8) = 1;
  auto trias = mesh.template get_elements<numeric::mesh::Tria<1>>();
  trias(0, 0) = 0;
  trias(1, 0) = 1;
  trias(2, 0) = 4;
  trias(0, 1) = 0;
  trias(1, 1) = 4;
  trias(2, 1) = 3;
  trias(0, 2) = 4;
  trias(1, 2) = 5;
  trias(2, 2) = 8;
  trias(0, 3) = 4;
  trias(1, 3) = 8;
  trias(2, 3) = 7;
  auto quads = mesh.template get_elements<numeric::mesh::Quad<1>>();
  quads(0, 0) = 1;
  quads(1, 0) = 2;
  quads(2, 0) = 5;
  quads(3, 0) = 4;
  quads(0, 1) = 3;
  quads(1, 1) = 4;
  quads(2, 1) = 7;
  quads(3, 1) = 6;
  return mesh;
}

TEST(mesh, subelement_relation_2d_points) {
  const auto mesh = generate_mesh();
  const auto [points, relations] =
      numeric::mesh::subelement_relation<numeric::mesh::Point<1>>(mesh);
  ASSERT_EQ(points.shape(0), 1);
  ASSERT_EQ(points.shape(1), 9);
  ASSERT_EQ((points(0, 0)), 0);
  ASSERT_EQ((points(0, 1)), 1);
  ASSERT_EQ((points(0, 2)), 4);
  ASSERT_EQ((points(0, 3)), 3);
  ASSERT_EQ((points(0, 4)), 5);
  ASSERT_EQ((points(0, 5)), 8);
  ASSERT_EQ((points(0, 6)), 7);
  ASSERT_EQ((points(0, 7)), 2);
  ASSERT_EQ((points(0, 8)), 6);
  const auto &trias = relations.template get<numeric::mesh::Tria<1>>();
  ASSERT_EQ((trias(0, 0)), 0);
  ASSERT_EQ((trias(0, 1)), 0);
  ASSERT_EQ((trias(0, 2)), 2);
  ASSERT_EQ((trias(0, 3)), 2);
  ASSERT_EQ((trias(1, 0)), 1);
  ASSERT_EQ((trias(1, 1)), 2);
  ASSERT_EQ((trias(1, 2)), 4);
  ASSERT_EQ((trias(1, 3)), 5);
  ASSERT_EQ((trias(2, 0)), 2);
  ASSERT_EQ((trias(2, 1)), 3);
  ASSERT_EQ((trias(2, 2)), 5);
  ASSERT_EQ((trias(2, 3)), 6);
  const auto &quads = relations.template get<numeric::mesh::Quad<1>>();
  ASSERT_EQ((quads(0, 0)), 1);
  ASSERT_EQ((quads(0, 1)), 3);
  ASSERT_EQ((quads(1, 0)), 7);
  ASSERT_EQ((quads(1, 1)), 2);
  ASSERT_EQ((quads(2, 0)), 4);
  ASSERT_EQ((quads(2, 1)), 6);
  ASSERT_EQ((quads(3, 0)), 2);
  ASSERT_EQ((quads(3, 1)), 8);
}

TEST(mesh, subelement_relation_2d_segments) {
  const auto mesh = generate_mesh();
  const auto [segments, relations] =
      numeric::mesh::subelement_relation<numeric::mesh::Segment<1>>(mesh);
  ASSERT_EQ(segments.shape(0), 2);
  ASSERT_EQ(segments.shape(1), 14);
  ASSERT_EQ((segments(0, 0)), 0);
  ASSERT_EQ((segments(1, 0)), 1);
  ASSERT_EQ((segments(0, 1)), 1);
  ASSERT_EQ((segments(1, 1)), 4);
  ASSERT_EQ((segments(0, 2)), 4);
  ASSERT_EQ((segments(1, 2)), 0);
  ASSERT_EQ((segments(0, 3)), 4);
  ASSERT_EQ((segments(1, 3)), 3);
  ASSERT_EQ((segments(0, 4)), 3);
  ASSERT_EQ((segments(1, 4)), 0);
  ASSERT_EQ((segments(0, 5)), 4);
  ASSERT_EQ((segments(1, 5)), 5);
  ASSERT_EQ((segments(0, 6)), 5);
  ASSERT_EQ((segments(1, 6)), 8);
  ASSERT_EQ((segments(0, 7)), 8);
  ASSERT_EQ((segments(1, 7)), 4);
  ASSERT_EQ((segments(0, 8)), 8);
  ASSERT_EQ((segments(1, 8)), 7);
  ASSERT_EQ((segments(0, 9)), 7);
  ASSERT_EQ((segments(1, 9)), 4);
  ASSERT_EQ((segments(0, 10)), 1);
  ASSERT_EQ((segments(1, 10)), 2);
  ASSERT_EQ((segments(0, 11)), 2);
  ASSERT_EQ((segments(1, 11)), 5);
  ASSERT_EQ((segments(0, 12)), 6);
  ASSERT_EQ((segments(1, 12)), 7);
  ASSERT_EQ((segments(0, 13)), 3);
  ASSERT_EQ((segments(1, 13)), 6);
  const auto &trias = relations.template get<numeric::mesh::Tria<1>>();
  ASSERT_EQ((trias(0, 0)), 0);
  ASSERT_EQ((trias(0, 1)), 2);
  ASSERT_EQ((trias(0, 2)), 5);
  ASSERT_EQ((trias(0, 3)), 7);
  ASSERT_EQ((trias(1, 0)), 1);
  ASSERT_EQ((trias(1, 1)), 3);
  ASSERT_EQ((trias(1, 2)), 6);
  ASSERT_EQ((trias(1, 3)), 8);
  ASSERT_EQ((trias(2, 0)), 2);
  ASSERT_EQ((trias(2, 1)), 4);
  ASSERT_EQ((trias(2, 2)), 7);
  ASSERT_EQ((trias(2, 3)), 9);
  const auto &quads = relations.template get<numeric::mesh::Quad<1>>();
  ASSERT_EQ((quads(0, 0)), 10);
  ASSERT_EQ((quads(0, 1)), 3);
  ASSERT_EQ((quads(1, 0)), 11);
  ASSERT_EQ((quads(1, 1)), 9);
  ASSERT_EQ((quads(2, 0)), 5);
  ASSERT_EQ((quads(2, 1)), 12);
  ASSERT_EQ((quads(3, 0)), 1);
  ASSERT_EQ((quads(3, 1)), 13);
}
