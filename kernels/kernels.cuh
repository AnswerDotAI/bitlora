// Kernel implementations
//
// @avh

#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <bit>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#ifndef DEBUG
#define DEBUG 1
#include "kernel_tools.cuh"
#endif

static constexpr int kTileWidth = 32;

/**
  Basic tiled 1-bit HQQ dequant and matmul
  Invoked with a 2D grid of 2D blocks
*/
__global__ void mm1bv1(const unsigned char *w, const float *z, const float *s,
                       const float *xin, float *out, int M, int K, int N,
                       int GS) {

  __shared__ float xTile[kTileWidth * kTileWidth];
  __shared__ unsigned char wTile[kTileWidth * kTileWidth / 8 + 1];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  float p = 0.0;

  for (int ph = 0; ph < (K + kTileWidth - 1) / kTileWidth; ph++) {
    if (x < N && ph * kTileWidth + ty < K) {
      xTile[ty * kTileWidth + tx] = xin[(ph * kTileWidth + ty) * N + x];
    } else {
      xTile[ty * kTileWidth + tx] = 0.0;
    }

    if (ph * kTileWidth + tx < K && y < M) {
      if (tx % 8 == 0) {
        wTile[ty * kTileWidth / 8 + tx / 8] =
            w[(y * K + (ph * kTileWidth + tx)) / 8];

#if DEBUG
        int row_bit = ph * kTileWidth + tx;
        int ti = ty * kTileWidth / 8 + tx / 8;
        printf("Writing values to smem (bit space): (%d, %d-%d) "
               "%c%c%c%c%c%c%c%c written to (%d x %d + %d) \n",
               y, row_bit, row_bit + 8, w[ti] >> 0 & 0b1 ? 'X' : '.',
               w[ti] >> 1 & 0b1 ? 'X' : '.', w[ti] >> 2 & 0b1 ? 'X' : '.',
               w[ti] >> 3 & 0b1 ? 'X' : '.', w[ti] >> 4 & 0b1 ? 'X' : '.',
               w[ti] >> 5 & 0b1 ? 'X' : '.', w[ti] >> 6 & 0b1 ? 'X' : '.',
               w[ti] >> 7 & 0b1 ? 'X' : '.', ty, kTileWidth, tx / 8);
#endif // DEBUG
      }
    } else {
      wTile[ty * kTileWidth / 8 + tx / 8] = 0;
    }

    __syncthreads();

#if DEBUG

    // block to check for debugging
    static constexpr int xb = 0;
    static constexpr int yb = 0;

    if (blockIdx.x == xb && blockIdx.y == yb && tx == 0 && ty == 0) {
      printf("\nBlock %d %d, W Tile %d\n", blockIdx.x, blockIdx.y, ph);
      show_tile1b(wTile, kTileWidth, kTileWidth);
      printf("\nBlock %d %d, x Tile %d\n", blockIdx.x, blockIdx.y, ph);
      show_tilef(xTile, kTileWidth, kTileWidth);
    }
    __syncthreads();
#endif // DEBUG

    for (int k = 0; k < kTileWidth; k++) {
      size_t woffset = ty * kTileWidth + k;
      size_t xoffset = k * kTileWidth + tx;
      int bit_value = (wTile[woffset / 8] >> (woffset % 8)) & 0b1;
      if (y < M && x < N && k < K) {
        int group = woffset / GS;
        float scale = s[group];
        float zp = z[group];
        // TODO: would this be faster as a multiplication?
        if (bit_value == 1) {
          p += xTile[xoffset] * scale - scale * zp;
        } else {
          p -= scale * zp;
        }
      }
    }

    __syncthreads();
  }

  if (y < M && x < N) {
    out[y * N + x] = p;
  }
}

/**
  CuTE implementation of the 1-bit HQQ dequant and matmul
  Invoked with 1D grid of blocks
  TODO: WIP / not finished

*/
__global__ void mm1bv2(const unsigned char *w, const float *z, const float *s,
                       const float *xin, float *out, int M, int K, int N,
                       int GS) {

  // Layout for uncompressed W
  cute::Layout wLayout = cute::make_layout(M, K);

  cute::Layout xLayout = cute::make_layout(K, N);
  cute::Layout outLayout = cute::make_layout(M, N);
  cute::Layout sLayout = cute::make_layout(M * K / GS);
  cute::Layout zLayout = cute::make_layout(M * K / GS);
  cute::Tensor wTensor = cute::make_tensor(cute::make_gmem_ptr(w), wLayout);
  // cute::Tensor test = make_tensor(mke_gmem_ptr(z),
}

/**
  HQQ style 3bit / 32 bit packing.
  Unlike the 1-bit kernel, we use 1D blocks ala SB's tiled matmul
  implementation.
  TODO: s/z dequant not implemented
 */
template <size_t BLOCK_SIZE>
__global__ void mm3bv1(const uint32_t *w, const float *x, float *out, int M,
                       int K, int N) {

  // Tile index for this thread's row in A and column in B
  // These values should be the same for all threads working with the same
  // shared memory.
  const uint tileRow = blockIdx.y; // ranges from 0 to M / BLOCK_SIZE
  const uint tileCol = blockIdx.x; // ranges from 0 to N / BLOCK_SIZE

  static constexpr size_t kBitsPerVal = 10;

  __shared__ float wTile[kTileWidth * kTileWidth];
  __shared__ float xTile[kTileWidth * kTileWidth];

  // pointer into W, x, out
  // size_t wPtr = tileRow * BLOCK_SIZE * K;
  size_t wPtr = tileRow * BLOCK_SIZE * K / kBitsPerVal;
  size_t xPtr = tileCol * BLOCK_SIZE;
  size_t outPtr = /* row offset = */ (tileRow * BLOCK_SIZE) * N +
                  /* col offset = */ tileCol * BLOCK_SIZE;

  // row and column within a tile for this thread
  const uint threadRow = threadIdx.x / BLOCK_SIZE;
  const uint threadCol = threadIdx.x % BLOCK_SIZE;

#if DEBUG
  static constexpr int xb = 0;
  static constexpr int yb = 0;
#endif

  float total = 0.0;
  // Within tileRow of A, tile iteration moves left to right
  // within tileCol of B, tile iteration moves top to bottom
  for (int tileIdx = 0; tileIdx < K / 10; tileIdx += BLOCK_SIZE) {
    // Write to shared memory
    const uint8_t shift_amount = (
        /* bits per uint32_t value */ 32 -
        /* 00 padding */ 2 -
        /* data bits */ 3 -
        /* position within 32 bits */ (threadCol % 10) * 3);

#if DEBUG
    if (blockIdx.x == xb && blockIdx.y == yb && threadIdx.x == 0 &&
        threadIdx.y == 0) {
      printf("shift amount: %d\n", shift_amount);
      printf("w value: %u\n", w[wPtr + threadRow * K / kBitsPerVal + threadCol / kBitsPerVal]);
      uint32_t dummy = 
        (w[wPtr + threadRow * K / kBitsPerVal + threadCol / kBitsPerVal] &
        (0b11 << shift_amount)) >> shift_amount;
      printf("Extracted value: %u\n", dummy);
    }
#endif

    // TODO: add dequant to wTile value
    wTile[threadRow * BLOCK_SIZE + threadCol] = static_cast<float>(
        (w[wPtr + threadRow * K / kBitsPerVal + threadCol / kBitsPerVal] &
        (0b11 << shift_amount)) >> shift_amount);
    xTile[threadRow * BLOCK_SIZE + threadCol] =
        x[xPtr + threadRow * N + threadCol];

#if DEBUG
    if (blockIdx.x == xb && blockIdx.y == yb && threadIdx.x == 0 &&
        threadIdx.y == 0) {
      // showd(wTile, kTileWidth, kTileWidth, "W Tile");
      showd(w, M, K/10, "W");
      showd(wTile, kTileWidth, kTileWidth, "W Tile");
    }
#endif

    __syncthreads();

    wPtr += BLOCK_SIZE / kBitsPerVal;
    xPtr += BLOCK_SIZE * N;

    // Multiply the two shared memory matrices
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      total +=
          wTile[threadRow * BLOCK_SIZE + k] * xTile[k * BLOCK_SIZE + threadCol];
    }

    __syncthreads();
  }
  if (tileRow * BLOCK_SIZE + threadRow < M &&
      tileCol * BLOCK_SIZE + threadCol < N) {
    out[outPtr + threadRow * N + threadCol] = total;
    // out[outPtr + threadRow * N + threadCol] = -1.0; // TODO: remove test value
  }
}

#endif // KERNELS_CUH
