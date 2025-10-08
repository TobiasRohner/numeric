/**
 * @page fd_poisson_tutorial Finite-Difference Poisson Solver Using Conjugate
 * Gradient
 *
 * @section intro_sec Introduction
 *
 * This example demonstrates how to solve a finite-difference approximation
 * to the Poisson problem using a conjugate gradient method. In this example,
 * we:
 *
 * - Define a regular grid and compute cell positions.
 * - Compute an exact solution (for testing purposes).
 * - Build the right-hand side and initialize an approximate solution.
 * - Define a negative Laplacian operator using finite differences.
 * - Solve the system using a conjugate gradient solver.
 * - Compute error metrics.
 * - Copy data to the host and write output to a NetCDF file.
 *
 * The complete source code is provided below:
 *
 * @snippet fd_poisson.cpp full_example
 *
 * @section breakdown_sec Code Breakdown
 *
 * The code is broken down into the following detailed sections:
 *
 * @subsection setup_sec 1. Setup and Grid Definition
 *
 * Define the memory type, grid size, tolerance for the solver, and create
 * a regular grid. The grid is created with a specified origin and size.
 * Cell positions are computed to be used later for the exact solution.
 *
 * @snippet fd_poisson.cpp setup
 *
 * @subsection exact_solution_sec 2. Define the Exact Solution and RHS
 *
 * The exact solution is given by a simple function of the cell coordinates
 * (e.g., \(u_{exact}(x,y) = x^2 + y^2\)). We also initialize the right-hand
 * side
 * \(f\) as a constant array (here \(-4\)), which is typical for a Laplacian
 * operator.
 *
 * @snippet fd_poisson.cpp exact_solution
 *
 * @subsection initialization_sec 3. Initialize the Approximate Solution
 *
 * We initialize the solution array with the exact values, and then set the
 * interior (non-boundary) nodes to zero as the starting guess for the solver.
 *
 * @snippet fd_poisson.cpp initialization
 *
 * @subsection laplacian_sec 4. Define the Negative Laplacian Operator
 *
 * A lambda function is used to define the negative Laplacian operator via
 * finite differences. The operator takes an array view as input and returns
 * the discrete Laplacian approximation. A temporary array (named `bd`) is
 * used to handle the boundary conditions.
 *
 * @snippet fd_poisson.cpp laplacian
 *
 * @subsection solver_sec 5. Conjugate Gradient Solver
 *
 * We create an instance of the conjugate gradient solver by passing the
 * negative Laplacian operator. The tolerance and maximum iterations are set.
 * The solver is then applied on the interior of the grid to update the
 * solution.
 *
 * @snippet fd_poisson.cpp solver
 *
 * @subsection output_sec 6. Output and Error Analysis
 *
 * The code computes the L2 norm error between the computed and exact solution.
 * Afterwards, the data is copied from device (or internal) memory to host
 * arrays, and a NetCDF file is created to write the arrays for further
 * inspection or visualization.
 *
 * @snippet fd_poisson.cpp output
 */

//! [full_example]
#include <iostream>
#include <numeric/io/netcdf_file.hpp>
#include <numeric/math/conjugate_gradient.hpp>
#include <numeric/math/sum.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_op.hpp>
#include <numeric/memory/constant.hpp>
#include <numeric/memory/linspace.hpp>
#include <numeric/memory/meshgrid.hpp>
#include <numeric/mesh/regular_grid.hpp>

using namespace numeric;

using real_t = float;

int main() {

  //! [setup]
  // Define the memory type (using host memory in this example)
  const memory::MemoryType memory_type = memory::MemoryType::HOST;

  // Set the grid dimension and solver tolerance
  static constexpr dim_t N = 1024;
  const real_t tol = 1e-5;

  // Create a regular grid with shape N x N. The grid spans a 2x2 area with an
  // origin at (-1,-1)
  mesh::RegularGrid<real_t, 2> mesh(memory::Shape<2>(N, N), 1, memory_type);
  mesh.set_origin(-1, -1);
  mesh.set_size(2, 2);

  // Compute the grid spacing in both directions
  const real_t dx = mesh.dx(0);
  const real_t dy = mesh.dx(1);

  // Get the cell positions for each grid point. These will be used to compute
  // the exact solution.
  const auto [X, Y] = mesh.cell_positions();
  //! [setup]

  //! [exact_solution]
  // Define the exact solution u_exact = X^2 + Y^2 for validation purposes.
  const memory::Array<real_t, 2> u_exact = X * X + Y * Y;

  // Construct the right-hand side f as a constant array filled with -4.
  const memory::Array<real_t, 2> f = memory::Constant<real_t, 2>(
      memory::Shape<2>(N + 2, N + 2), -4, memory_type);
  //! [exact_solution]

  //! [initialization]
  // Allocate the solution array u with ghost cells (size: (N+2) x (N+2)) and
  // initialize it with u_exact.
  memory::Array<real_t, 2> u(memory::Shape<2>(N + 2, N + 2), memory_type);
  u = u_exact;
  // Set the interior of u (excluding boundaries) to zero, as the initial guess
  // for the solver.
  u(memory::Slice(1, -2), memory::Slice(1, -2)) = 0;
  //! [initialization]

  //! [laplacian]
  // Create a temporary array "bd" to facilitate the finite-difference
  // computation.
  memory::Array<real_t, 2> bd = u_exact;
  // Define the negative Laplacian operator as a lambda function.
  // The operator computes a finite-difference approximation of the Laplacian.
  auto negative_laplacian = [&](memory::ArrayConstView<real_t, 2> &x) {
    bd(memory::Slice(1, -2), memory::Slice(1, -2)) = x;
    return (4 * bd(memory::Slice(1, -2), memory::Slice(1, -2)) -
            bd(memory::Slice(1, -2), memory::Slice(0, -3)) -
            bd(memory::Slice(1, -2), memory::Slice(2, -1)) -
            bd(memory::Slice(0, -3), memory::Slice(1, -2)) -
            bd(memory::Slice(2, -1), memory::Slice(1, -2))) /
           (dx * dy);
  };
  //! [laplacian]

  //! [solver]
  // Create the conjugate gradient solver using the negative Laplacian operator.
  math::ConjugateGradient<real_t, decltype(negative_laplacian)> cg(
      negative_laplacian);
  cg.set_tolerance(N * N * 1e-8);
  cg.set_max_iterations(N * N);

  // Solve the system on the interior of the grid.
  const auto [converged, num_iter, error] =
      cg.solve(f(memory::Slice(1, -2), memory::Slice(1, -2)),
               u(memory::Slice(1, -2), memory::Slice(1, -2)));
  if (converged) {
    std::cout << "Converged in " << num_iter << " iterations (error = " << error
              << ")" << std::endl;
  } else {
    std::cout << "No convergence in " << num_iter
              << " iterations (error = " << error << ")" << std::endl;
  }
  std::cout << "L2-err = " << math::norm::l2(u - u_exact) / (N * N)
            << std::endl;
  //! [solver]

  //! [output]
  // Copy arrays to host for writing to file.
  memory::Array<real_t, 2> f_host(f.shape());
  memory::Array<real_t, 2> u_host(u.shape());
  memory::Array<real_t, 2> u_exact_host(u_exact.shape());
  memory::copy(f_host, f);
  memory::copy(u_host, u);
  memory::copy(u_exact_host, u_exact);

  // Create a NetCDF file and write the arrays.
  auto file = io::NetCDFFile::create("fd_poisson.nc");
  const auto dim_N = file->create_dim("N", N + 2);
  file->write("f", f_host, {dim_N, dim_N});
  file->write("u", u_host, {dim_N, dim_N});
  file->write("u_exact", u_exact_host, {dim_N, dim_N});
  //! [output]

  return 0;
}
//! [full_example]
