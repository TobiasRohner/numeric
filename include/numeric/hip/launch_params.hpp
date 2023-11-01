#ifndef NUMERIC_HIP_LAUNCH_PARAMS_HPP_
#define NUMERIC_HIP_LAUNCH_PARAMS_HPP_

#ifndef __HIP_DEVICE_COMPILE__
#include <iostream>
#endif

namespace numeric::hip {

struct LaunchParams {
  unsigned grid_dim_x, grid_dim_y, grid_dim_z;
  unsigned block_dim_x, block_dim_y, block_dim_z;
  unsigned shared_mem_bytes = 0;
};

#ifndef __HIP_DEVICE_COMPILE__
std::ostream &operator<<(std::ostream &os, const LaunchParams &lp);
#endif

} // namespace numeric::hip

#endif
