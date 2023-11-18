#ifndef NUMERIC_HIP_DEVICE_HPP_
#define NUMERIC_HIP_DEVICE_HPP_

#include <numeric/hip/launch_params.hpp>

namespace numeric::hip {

class Device {
public:
  Device();
  Device(int id);
  Device(const Device &) = default;
  Device &operator=(const Device &) = default;

  static int count();

  void activate() const;
  void sync() const;

  int id() const;

  unsigned max_block_dim_x() const;
  unsigned max_block_dim_y() const;
  unsigned max_block_dim_z() const;
  unsigned max_grid_dim_x() const;
  unsigned max_grid_dim_y() const;
  unsigned max_grid_dim_z() const;
  unsigned max_threads_per_block() const;

  LaunchParams launch_params_for_grid(unsigned Nx, unsigned Ny,
                                      unsigned Nz) const;

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
