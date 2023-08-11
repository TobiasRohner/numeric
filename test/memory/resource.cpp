#include <gtest/gtest.h>

#include <numeric/memory/host_memory_resource.hpp>
#if NUMERIC_ENABLE_HIP
#include <numeric/memory/pinned_memory_resource.hpp>
#include <numeric/memory/device_memory_resource.hpp>
#endif



TEST(memory, resource_host) {
  numeric::memory::HostMemoryResource<double> resource;
  double *ptr = resource.allocate(128);
  ASSERT_NE(ptr, nullptr);
  resource.deallocate(ptr, 128);
}


#if NUMERIC_ENABLE_HIP
TEST(memory, resource_pinned) {
  numeric::memory::PinnedMemoryResource<double> resource;
  double *ptr = resource.allocate(128);
  ASSERT_NE(ptr, nullptr);
  resource.deallocate(ptr, 128);
}

TEST(memory, resource_device) {
  numeric::memory::DeviceMemoryResource<double> resource;
  double *ptr = resource.allocate(128);
  ASSERT_NE(ptr, nullptr);
  resource.deallocate(ptr, 128);
}
#endif
