/**
 * @page fd_wave_tutorial Finite-Difference Wave Equation Simulation
 *
 * @section intro_sec Introduction
 *
 * This example simulates a one-dimensional wave equation using a
 * finite-difference method. The wave equation being solved is:
 *
 * @f[
 *   \frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}
 * @f]
 *
 * where @f$u(x,t)@f$ is the wave displacement, and @f$c@f$ is the wave speed.
 *
 * The Numeric library is used for:
 * - Managing memory arrays with rich operator overloads
 * - Applying element-wise operations
 * - Performing slicing operations on arrays
 * - Outputting data to a NetCDF file for visualization and post-processing.
 *
 * The complete source code is shown below:
 *
 * @snippet fd_wave.cpp full_example
 *
 * @section breakdown_sec Code Breakdown
 *
 * The code is broken down into the following sections:
 *
 * @subsection simulation_parameters_sec Simulation Parameters
 *
 * We begin by defining the simulation parameters. This includes the spatial
 * domain limits, the total simulation time, the Courant number (for stability),
 * and the number of spatial grid points. The spatial and temporal resolution
 * are derived from these parameters.
 *
 * @snippet fd_wave.cpp simulation_parameters
 *
 * @subsection memory_allocation_sec Memory Allocation and Initialization
 *
 * Three 2D arrays are allocated using the Numeric memory abstractions:
 * - **u**: The wave displacement.
 * - **v**: The wave velocity.
 * - **a**: The wave acceleration.
 *
 * Their dimensions are set to (M+1) x N, where each row corresponds to a time
 * step and each column corresponds to a spatial grid point. The arrays are
 * first initialized to zero. The initial condition for the displacement is set
 * to a Gaussian profile.
 *
 * @snippet fd_wave.cpp memory_allocation
 *
 * @subsection time_stepping_sec Finite-Difference Time-Stepping Loop
 *
 * The core of the simulation is a time-stepping loop. In each step:
 *
 * - The acceleration is computed as the finite-difference approximation of the
 * second spatial derivative of the displacement.
 * - The velocity is updated using the computed acceleration.
 * - The displacement is then updated using the new velocity.
 *
 * Only the interior points (from 1 to N-2) are updated because the boundary
 * points are typically handled by fixed boundary conditions.
 *
 * @snippet fd_wave.cpp time_stepping
 *
 * @subsection output_sec Writing the Output to NetCDF
 *
 * After the simulation completes, the computed arrays for displacement,
 * velocity, and acceleration are written to a NetCDF file. This allows for easy
 * post-processing and visualization.
 *
 * @snippet fd_wave.cpp output_netcdf
 */

//! [full_example]
#include <numeric/io/netcdf_file.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_op.hpp>
#include <numeric/memory/copyer.hpp>
#include <numeric/memory/linspace.hpp>
#if NUMERIC_ENABLE_HIP
#include <numeric/hip/kernel.hpp>
#endif

using namespace numeric;

int main() {

  //! [simulation_parameters]
  // Simulation domain and parameters
  const double x0 = -1; ///< Left boundary of the spatial domain.
  const double x1 = 1;  ///< Right boundary of the spatial domain.
  const double T = 10;  ///< Total simulation time.
  const double c = 1;   ///< Wave speed.
  const dim_t N = 5000; ///< Number of spatial grid points.
  const double C = 0.5; ///< Courant number (stability factor).
  const double dx = (x1 - x0) / (N - 1); ///< Spatial step size.
  const double dt = C * dx / c; ///< Time step size based on the CFL condition.
  const dim_t M = T / dt;       ///< Number of time steps.
  const memory::MemoryType memory_type = memory::MemoryType::HOST;
  //! [simulation_parameters]

  //! [memory_allocation]
  // Allocate 2D arrays for displacement (u), velocity (v), and acceleration
  // (a). Each array has (M+1) rows (one for each time step) and N columns (one
  // for each spatial point).
  memory::Array<double, 2> u(memory::Shape<2>(M + 1, N), memory_type);
  memory::Array<double, 2> v(memory::Shape<2>(M + 1, N), memory_type);
  memory::Array<double, 2> a(memory::Shape<2>(M + 1, N), memory_type);

  // Initialize all arrays to zero.
  u = 0;
  v = 0;
  a = 0;

  // Set the initial condition for u using a Gaussian profile.
  const double sigma = 0.1; // Standard deviation for the Gaussian.
  u(0) = memory::exp(-memory::pow<2>(memory::linspace(x0, x1, N) / sigma));
  //! [memory_allocation]

  //! [time_stepping]
  // Time-stepping loop: Update acceleration, velocity, and displacement for
  // each time step.
  for (dim_t t = 1; t <= M; ++t) {
    // Compute acceleration using the finite-difference approximation for the
    // second derivative.
    a(t, memory::Slice(1, N - 1)) =
        c * c *
        (u(t - 1, memory::Slice(2, N)) + u(t - 1, memory::Slice(0, N - 2)) -
         2 * u(t - 1, memory::Slice(1, N - 1))) /
        (dx * dx);

    // Update velocity by integrating the acceleration.
    v(t, memory::Slice(1, N - 1)) =
        v(t - 1, memory::Slice(1, N - 1)) + dt * a(t, memory::Slice(1, N - 1));

    // Update displacement by integrating the velocity.
    u(t, memory::Slice(1, N - 1)) =
        u(t - 1, memory::Slice(1, N - 1)) + dt * v(t, memory::Slice(1, N - 1));
  }
  //! [time_stepping]

  //! [output_netcdf]
  // Create a NetCDF file to store the simulation results.
  auto file = io::NetCDFFile::create("fd_wave.nc");

  // Define NetCDF dimensions for time (M+1) and space (N).
  const auto dim_N = file->create_dim("N", N);
  const auto dim_M = file->create_dim("M", M + 1);

  // Write the arrays for displacement (u), velocity (v), and acceleration (a)
  // to the file.
  file->write("u", u, {dim_M, dim_N});
  file->write("v", v, {dim_M, dim_N});
  file->write("a", a, {dim_M, dim_N});
  //! [output_netcdf]

  return 0;
}
//! [full_example]
