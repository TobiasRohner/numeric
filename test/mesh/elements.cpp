#include <gtest/gtest.h>

#include <numeric/mesh/elements.hpp>

TEST(mesh, J_segment_1d) {
  using element_t = numeric::mesh::Segment<1>;
  static constexpr numeric::dim_t world_dim = 1;
  static constexpr double nodes_ref[world_dim][element_t::num_nodes] = {{0, 1}};
  static constexpr double x[element_t::dim] = {0.5};
  double J[world_dim][element_t::dim];
  element_t::jacobian(nodes_ref, x, J, world_dim);
  ASSERT_EQ(J[0][0], 1);
  static constexpr double nodes_scal[world_dim][element_t::num_nodes] = {
      {0, 2}};
  element_t::jacobian(nodes_scal, x, J, world_dim);
  ASSERT_EQ(J[0][0], 2);
}

TEST(mesh, J_segment_2d) {
  using element_t = numeric::mesh::Segment<1>;
  static constexpr numeric::dim_t world_dim = 2;
  static constexpr double nodes_ref[world_dim][element_t::num_nodes] = {{0, 1},
                                                                        {0, 0}};
  static constexpr double x[element_t::dim] = {0.5};
  double J[world_dim][element_t::dim];
  element_t::jacobian(nodes_ref, x, J, world_dim);
  ASSERT_EQ(J[0][0], 1);
  ASSERT_EQ(J[1][0], 0);
  static constexpr double nodes_scal[world_dim][element_t::num_nodes] = {
      {0, 2}, {0, 0}};
  element_t::jacobian(nodes_scal, x, J, world_dim);
  ASSERT_EQ(J[0][0], 2);
  ASSERT_EQ(J[1][0], 0);
}

TEST(mesh, J_segment_3d) {
  using element_t = numeric::mesh::Segment<1>;
  static constexpr numeric::dim_t world_dim = 3;
  static constexpr double nodes_ref[world_dim][element_t::num_nodes] = {
      {0, 1}, {0, 0}, {0, 0}};
  static constexpr double x[element_t::dim] = {0.5};
  double J[world_dim][element_t::dim];
  element_t::jacobian(nodes_ref, x, J, world_dim);
  ASSERT_EQ(J[0][0], 1);
  ASSERT_EQ(J[1][0], 0);
  ASSERT_EQ(J[2][0], 0);
  static constexpr double nodes_scal[world_dim][element_t::num_nodes] = {
      {0, 2}, {0, 0}, {0, 0}};
  element_t::jacobian(nodes_scal, x, J, world_dim);
  ASSERT_EQ(J[0][0], 2);
  ASSERT_EQ(J[1][0], 0);
  ASSERT_EQ(J[2][0], 0);
}

TEST(mesh, J_tria_2d) {
  using element_t = numeric::mesh::Tria<1>;
  static constexpr numeric::dim_t world_dim = 2;
  static constexpr double nodes_ref[world_dim][element_t::num_nodes] = {
      {0, 1, 0}, {0, 0, 1}};
  static constexpr double x[element_t::dim] = {1. / 3, 1. / 3};
  double J[world_dim][element_t::dim];
  element_t::jacobian(nodes_ref, x, J, world_dim);
  ASSERT_EQ(J[0][0], 1);
  ASSERT_EQ(J[0][1], 0);
  ASSERT_EQ(J[1][0], 0);
  ASSERT_EQ(J[1][1], 1);
  static constexpr double nodes_scal[world_dim][element_t::num_nodes] = {
      {0, 2, 0}, {0, 0, 3}};
  element_t::jacobian(nodes_scal, x, J, world_dim);
  ASSERT_EQ(J[0][0], 2);
  ASSERT_EQ(J[0][1], 0);
  ASSERT_EQ(J[1][0], 0);
  ASSERT_EQ(J[1][1], 3);
}

TEST(mesh, J_tria_3d) {
  using element_t = numeric::mesh::Tria<1>;
  static constexpr numeric::dim_t world_dim = 3;
  static constexpr double nodes_ref[world_dim][element_t::num_nodes] = {
      {0, 1, 0}, {0, 0, 1}, {0, 0, 0}};
  static constexpr double x[element_t::dim] = {1. / 3, 1. / 3};
  double J[world_dim][element_t::dim];
  element_t::jacobian(nodes_ref, x, J, world_dim);
  ASSERT_EQ(J[0][0], 1);
  ASSERT_EQ(J[0][1], 0);
  ASSERT_EQ(J[1][0], 0);
  ASSERT_EQ(J[1][1], 1);
  ASSERT_EQ(J[2][0], 0);
  ASSERT_EQ(J[2][1], 0);
  static constexpr double nodes_scal[world_dim][element_t::num_nodes] = {
      {0, 2, 0}, {0, 0, 3}, {0, 0, 0}};
  element_t::jacobian(nodes_scal, x, J, world_dim);
  ASSERT_EQ(J[0][0], 2);
  ASSERT_EQ(J[0][1], 0);
  ASSERT_EQ(J[1][0], 0);
  ASSERT_EQ(J[1][1], 3);
  ASSERT_EQ(J[2][0], 0);
  ASSERT_EQ(J[2][1], 0);
}

TEST(mesh, J_quad_2d) {
  using element_t = numeric::mesh::Quad<1>;
  static constexpr numeric::dim_t world_dim = 2;
  static constexpr double nodes_ref[world_dim][element_t::num_nodes] = {
      {0, 1, 1, 0}, {0, 0, 1, 1}};
  static constexpr double x[element_t::dim] = {0.5, 0.5};
  double J[world_dim][element_t::dim];
  element_t::jacobian(nodes_ref, x, J, world_dim);
  ASSERT_EQ(J[0][0], 1);
  ASSERT_EQ(J[0][1], 0);
  ASSERT_EQ(J[1][0], 0);
  ASSERT_EQ(J[1][1], 1);
  static constexpr double nodes_scal[world_dim][element_t::num_nodes] = {
      {0, 2, 2, 0}, {0, 0, 3, 3}};
  element_t::jacobian(nodes_scal, x, J, world_dim);
  ASSERT_EQ(J[0][0], 2);
  ASSERT_EQ(J[0][1], 0);
  ASSERT_EQ(J[1][0], 0);
  ASSERT_EQ(J[1][1], 3);
}

TEST(mesh, J_quad_3d) {
  using element_t = numeric::mesh::Quad<1>;
  static constexpr numeric::dim_t world_dim = 3;
  static constexpr double nodes_ref[world_dim][element_t::num_nodes] = {
      {0, 1, 1, 0}, {0, 0, 1, 1}, {0, 0, 0, 0}};
  static constexpr double x[element_t::dim] = {0.5, 0.5};
  double J[world_dim][element_t::dim];
  element_t::jacobian(nodes_ref, x, J, world_dim);
  ASSERT_EQ(J[0][0], 1);
  ASSERT_EQ(J[0][1], 0);
  ASSERT_EQ(J[1][0], 0);
  ASSERT_EQ(J[1][1], 1);
  ASSERT_EQ(J[2][0], 0);
  ASSERT_EQ(J[2][1], 0);
  static constexpr double nodes_scal[world_dim][element_t::num_nodes] = {
      {0, 2, 2, 0}, {0, 0, 3, 3}, {0, 0, 0, 0}};
  element_t::jacobian(nodes_scal, x, J, world_dim);
  ASSERT_EQ(J[0][0], 2);
  ASSERT_EQ(J[0][1], 0);
  ASSERT_EQ(J[1][0], 0);
  ASSERT_EQ(J[1][1], 3);
  ASSERT_EQ(J[2][0], 0);
  ASSERT_EQ(J[2][1], 0);
}

TEST(mesh, J_tetra_3d) {
  using element_t = numeric::mesh::Tetra<1>;
  static constexpr numeric::dim_t world_dim = 3;
  static constexpr double nodes_ref[world_dim][element_t::num_nodes] = {
      {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  static constexpr double x[element_t::dim] = {0.25, 0.25, 0.25};
  double J[world_dim][element_t::dim];
  element_t::jacobian(nodes_ref, x, J, world_dim);
  ASSERT_EQ(J[0][0], 1);
  ASSERT_EQ(J[0][1], 0);
  ASSERT_EQ(J[0][2], 0);
  ASSERT_EQ(J[1][0], 0);
  ASSERT_EQ(J[1][1], 1);
  ASSERT_EQ(J[1][2], 0);
  ASSERT_EQ(J[2][0], 0);
  ASSERT_EQ(J[2][1], 0);
  ASSERT_EQ(J[2][2], 1);
  static constexpr double nodes_scal[world_dim][element_t::num_nodes] = {
      {0, 2, 0, 0}, {0, 0, 3, 0}, {0, 0, 0, 4}};
  element_t::jacobian(nodes_scal, x, J, world_dim);
  ASSERT_EQ(J[0][0], 2);
  ASSERT_EQ(J[0][1], 0);
  ASSERT_EQ(J[0][2], 0);
  ASSERT_EQ(J[1][0], 0);
  ASSERT_EQ(J[1][1], 3);
  ASSERT_EQ(J[1][2], 0);
  ASSERT_EQ(J[2][0], 0);
  ASSERT_EQ(J[2][1], 0);
  ASSERT_EQ(J[2][2], 4);
}

TEST(mesh, J_cube_3d) {
  using element_t = numeric::mesh::Cube<1>;
  static constexpr numeric::dim_t world_dim = 3;
  static constexpr double nodes_ref[world_dim][element_t::num_nodes] = {
      {0, 1, 1, 0, 0, 1, 1, 0},
      {0, 0, 1, 1, 0, 0, 1, 1},
      {0, 0, 0, 0, 1, 1, 1, 1}};
  static constexpr double x[element_t::dim] = {0.5, 0.5, 0.5};
  double J[world_dim][element_t::dim];
  element_t::jacobian(nodes_ref, x, J, world_dim);
  ASSERT_EQ(J[0][0], 1);
  ASSERT_EQ(J[0][1], 0);
  ASSERT_EQ(J[0][2], 0);
  ASSERT_EQ(J[1][0], 0);
  ASSERT_EQ(J[1][1], 1);
  ASSERT_EQ(J[1][2], 0);
  ASSERT_EQ(J[2][0], 0);
  ASSERT_EQ(J[2][1], 0);
  ASSERT_EQ(J[2][2], 1);
  static constexpr double nodes_scal[world_dim][element_t::num_nodes] = {
      {0, 2, 2, 0, 0, 2, 2, 0},
      {0, 0, 3, 3, 0, 0, 3, 3},
      {0, 0, 0, 0, 4, 4, 4, 4}};
  element_t::jacobian(nodes_scal, x, J, world_dim);
  ASSERT_EQ(J[0][0], 2);
  ASSERT_EQ(J[0][1], 0);
  ASSERT_EQ(J[0][2], 0);
  ASSERT_EQ(J[1][0], 0);
  ASSERT_EQ(J[1][1], 3);
  ASSERT_EQ(J[1][2], 0);
  ASSERT_EQ(J[2][0], 0);
  ASSERT_EQ(J[2][1], 0);
  ASSERT_EQ(J[2][2], 4);
}

TEST(mesh, ident_integ_elem_point_0d) {
  using element_t = numeric::mesh::Point<1>;
  const double ie =
      element_t::integration_element<double>(nullptr, nullptr, 0, nullptr);
  ASSERT_EQ(ie, 1);
}

TEST(mesh, ident_integ_elem_point_1d) {
  using element_t = numeric::mesh::Point<1>;
  static constexpr double nodes[1][1] = {{0}};
  const double ie =
      element_t::integration_element<double>(nodes, nullptr, 1, nullptr);
  ASSERT_EQ(ie, 1);
}

TEST(mesh, ident_integ_elem_point_2d) {
  using element_t = numeric::mesh::Point<1>;
  static constexpr double nodes[2][1] = {{0}, {0}};
  const double ie =
      element_t::integration_element<double>(nodes, nullptr, 1, nullptr);
  ASSERT_EQ(ie, 1);
}

TEST(mesh, ident_integ_elem_point_3d) {
  using element_t = numeric::mesh::Point<1>;
  static constexpr double nodes[3][1] = {{0}, {0}, {0}};
  const double ie =
      element_t::integration_element<double>(nodes, nullptr, 1, nullptr);
  ASSERT_EQ(ie, 1);
}

TEST(mesh, ident_integ_elem_segment_1d) {
  using element_t = numeric::mesh::Segment<1>;
  static constexpr numeric::dim_t world_dim = 1;
  static constexpr double nodes[world_dim][2] = {{0, 1}};
  static constexpr double x[1] = {0.5};
  double work[element_t::dim * element_t::dim + world_dim * element_t::dim];
  const double ie = element_t::integration_element(nodes, x, world_dim, work);
  ASSERT_EQ(ie, 1);
}

TEST(mesh, ident_integ_elem_segment_2d) {
  using element_t = numeric::mesh::Segment<1>;
  static constexpr numeric::dim_t world_dim = 2;
  static constexpr double nodes[world_dim][2] = {{0, 1}, {0, 0}};
  static constexpr double x[1] = {0.5};
  double work[element_t::dim * element_t::dim + world_dim * element_t::dim];
  const double ie = element_t::integration_element(nodes, x, world_dim, work);
  ASSERT_EQ(ie, 1);
}

TEST(mesh, ident_integ_elem_segment_3d) {
  using element_t = numeric::mesh::Segment<1>;
  static constexpr numeric::dim_t world_dim = 3;
  static constexpr double nodes[world_dim][2] = {{0, 1}, {0, 0}, {0, 0}};
  static constexpr double x[1] = {0.5};
  double work[element_t::dim * element_t::dim + world_dim * element_t::dim];
  const double ie = element_t::integration_element(nodes, x, world_dim, work);
  ASSERT_EQ(ie, 1);
}

TEST(mesh, ident_integ_elem_tria_2d) {
  using element_t = numeric::mesh::Tria<1>;
  static constexpr numeric::dim_t world_dim = 2;
  static constexpr double nodes[world_dim][3] = {{0, 1, 0}, {0, 0, 1}};
  static constexpr double x[2] = {1. / 3, 1. / 3};
  double work[element_t::dim * element_t::dim + world_dim * element_t::dim];
  const double ie = element_t::integration_element(nodes, x, world_dim, work);
  ASSERT_EQ(ie, 1);
}

TEST(mesh, ident_integ_elem_tria_3d) {
  using element_t = numeric::mesh::Tria<1>;
  static constexpr numeric::dim_t world_dim = 3;
  static constexpr double nodes[world_dim][3] = {
      {0, 1, 0}, {0, 0, 1}, {0, 0, 0}};
  static constexpr double x[2] = {1. / 3, 1. / 3};
  double work[element_t::dim * element_t::dim + world_dim * element_t::dim];
  const double ie = element_t::integration_element(nodes, x, world_dim, work);
  ASSERT_EQ(ie, 1);
}

TEST(mesh, ident_integ_elem_quad_2d) {
  using element_t = numeric::mesh::Quad<1>;
  static constexpr numeric::dim_t world_dim = 2;
  static constexpr double nodes[world_dim][4] = {{0, 1, 1, 0}, {0, 0, 1, 1}};
  static constexpr double x[2] = {0.5, 0.5};
  double work[element_t::dim * element_t::dim + world_dim * element_t::dim];
  const double ie = element_t::integration_element(nodes, x, world_dim, work);
  ASSERT_EQ(ie, 1);
}

TEST(mesh, ident_integ_elem_quad_3d) {
  using element_t = numeric::mesh::Quad<1>;
  static constexpr numeric::dim_t world_dim = 3;
  static constexpr double nodes[world_dim][4] = {
      {0, 1, 1, 0}, {0, 0, 1, 1}, {0, 0, 0, 0}};
  static constexpr double x[2] = {1. / 3, 1. / 3};
  double work[element_t::dim * element_t::dim + world_dim * element_t::dim];
  const double ie = element_t::integration_element(nodes, x, world_dim, work);
  ASSERT_EQ(ie, 1);
}

TEST(mesh, ident_integ_elem_tetra_3d) {
  using element_t = numeric::mesh::Tetra<1>;
  static constexpr numeric::dim_t world_dim = 3;
  static constexpr double nodes[world_dim][4] = {
      {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  static constexpr double x[3] = {0.25, 0.25, 0.25};
  double work[element_t::dim * element_t::dim + world_dim * element_t::dim];
  const double ie = element_t::integration_element(nodes, x, world_dim, work);
  ASSERT_EQ(ie, 1);
}

TEST(mesh, ident_integ_elem_cube_3d) {
  using element_t = numeric::mesh::Cube<1>;
  static constexpr numeric::dim_t world_dim = 3;
  static constexpr double nodes[world_dim][8] = {{0, 1, 1, 0, 0, 1, 1, 0},
                                                 {0, 0, 1, 1, 0, 0, 1, 1},
                                                 {0, 0, 0, 0, 1, 1, 1, 1}};
  static constexpr double x[3] = {0.5, 0.5, 0.5};
  double work[element_t::dim * element_t::dim + world_dim * element_t::dim];
  const double ie = element_t::integration_element(nodes, x, world_dim, work);
  ASSERT_EQ(ie, 1);
}
