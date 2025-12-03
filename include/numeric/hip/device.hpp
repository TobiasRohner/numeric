#ifndef NUMERIC_HIP_DEVICE_HPP_
#define NUMERIC_HIP_DEVICE_HPP_

#include <numeric/hip/launch_params.hpp>

namespace numeric::hip {

/**
 * @brief Class representing an HIP device.
 */
class Device {
public:
  /**
   * @brief Default constructor.
   *
   * Constructs a Device object representing the default device.
   */
  Device();

  /**
   * @brief Constructor with device ID.
   *
   * Constructs a Device object representing the specified device ID.
   *
   * @param id ID of the device.
   */
  Device(int id);

  Device(const Device &) = default;
  Device &operator=(const Device &) = default;

  /**
   * @brief Gets the number of HIP capable devices.
   *
   * @return Number of HIP capable devices.
   */
  static int count();

  /**
   * @brief Sets this device to be the current one.
   */
  void activate() const;

  /**
   * @brief Waits for all operations on the device to finish.
   */
  void sync() const;

  /**
   * @brief Gets the ID of the device.
   *
   * @return ID of the device.
   */
  int id() const;

  unsigned max_block_dim_x() const;
  unsigned max_block_dim_y() const;
  unsigned max_block_dim_z() const;
  unsigned max_grid_dim_x() const;
  unsigned max_grid_dim_y() const;
  unsigned max_grid_dim_z() const;
  unsigned max_threads_per_block() const;
  unsigned max_shared_memory_per_block() const;
  int warp_size() const;

  /**
   * @brief Calculates somewhat optimal launch parameters for a grid.
   *
   * @param Nx Size of the grid in the x-direction.
   * @param Ny Size of the grid in the y-direction.
   * @param Nz Size of the grid in the z-direction.
   * #param shared_mem_bytes Amount of shared memory per block.
   * @return Launch parameters for the grid.
   */
  LaunchParams launch_params_for_grid(unsigned Nx, unsigned Ny,
                                      unsigned Nz) const;

  /**
   * @brief Executes a function while the device is active.
   *
   * @tparam Func Type of the function.
   * @param func Function to execute.
   */
  template <typename Func> void do_while_active(const Func &func) const {
    Device current;
    activate();
    func();
    current.activate();
  }

private:
  int id_;
};

} // namespace numeric::hip

#endif
