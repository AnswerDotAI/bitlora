// Kernel implementations
//
// @avh

#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

#ifndef DEBUG
#define DEBUG 1
#endif

__device__ void show_tile1b(const unsigned char *W_q, int R, int C) {
  for (int r = 0; r < R; r++) {
    printf("%5d: ", r);
    for (int c = 0; c < C; c++) {
      int current_bit = (W_q[r * (C / 8) + c / 8] >> (c % 8)) & 0b1;
      if (current_bit == 1) {
        printf("X");
      } else {
        printf(".");
      }
    }
    printf("\n");
  }
}

__device__ void show_tilef(const float *W, int R, int C) {
  for (int r = 0; r < R; r++) {
    printf("%5d: ", r);
    for (int c = 0; c < C; c++) {
      printf("%2.0f ", W[r * C + c]);
    }
    printf("\n");
  }
}

static constexpr int kTileWidth = 32;

__global__ void mm1b(const unsigned char *w, const float *z, const float *s,
                     const float *xin, float *out, int M, int K, int N, int G) {

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
      if (y < M && x < N && k < K && bit_value) {
        // TODO: incorporate zp and scale
        // int group = woffset / 64;
        // float scale = s[group];
        // printf("scale %f\n", scale);
        // float zp = z[group];
        p += xTile[xoffset];
      }
    }

    __syncthreads();
  }

  if (y < M && x < N) {
    out[y * N + x] = p;
  }
}

#endif // KERNELS_CUH
