#include <numeric/hip/launch_params.hpp>

namespace numeric::hip {

std::ostream &operator<<(std::ostream &os, const LaunchParams &lp) {
  os << "LaunchParams { grid_dim=(" << lp.grid_dim_x << "," << lp.grid_dim_y
     << "," << lp.grid_dim_z << "), block_dim=(" << lp.block_dim_x << ","
     << lp.block_dim_y << "," << lp.block_dim_z
     << "), shared_mem_bytes=" << lp.shared_mem_bytes << " }";
  return os;
}

} // namespace numeric::hip
