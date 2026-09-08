#include <numeric/hip/runtime.hpp>

hipError_t hipOccupancyMaxPotentialBlockSize(int *gridSize, int *blockSize,
                                             hipFunction_t f,
                                             size_t dynSharedMemPerBlk,
                                             int blockSizeLimit) {
  return hipCUResultTohipError(cuOccupancyMaxPotentialBlockSize(
      gridSize, blockSize, f, nullptr, dynSharedMemPerBlk, blockSizeLimit));
}
