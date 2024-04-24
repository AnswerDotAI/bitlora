// Runner script 1-bit matrix multiplication kernels
//
// @avh

#define ANKERL_NANOBENCH_IMPLEMENT

#include <array>
#include <cassert>
#include <iostream>
#include <random>

#include "nanobench.h"
#include "spdlog/spdlog.h"
#include <cuda_runtime.h>

#include "kernel_tools.cuh"
#include "kernels.cuh"

template <size_t rows, size_t cols>
void bit_encoder(const std::array<int, rows * cols> &matrix,
                 std::array<unsigned char, cdiv(rows *cols, 8)> &bit_matrix) {
  static_assert(cols > 0 && rows > 0, "Matrix dimensions must be positive");
  static_assert(cols % 8 == 0, "Matrix columns must be divisible by 8");
  size_t outer_idx = 0;
  size_t inner_idx = 0;
  bit_matrix[0] = 0;
  for (int val : matrix) {
    assert(val == 0 || val == 1);
    bit_matrix[outer_idx] +=
        val == 1 ? (0b01 << inner_idx) : (0b00 << inner_idx);
    inner_idx++;
    if (inner_idx == 8) {
      // At every 8th bit:
      // - increment the outer index
      // - reset inner index to 0
      // - zero-out next byte
      outer_idx++;
      inner_idx = 0;
      if (outer_idx < cdiv(rows * cols, 8)) {
        bit_matrix[outer_idx] = 0;
      }
    }
  }
}

template <size_t rows, size_t cols>
void bit_decoder(std::array<int, rows * cols> &matrix,
                 std::array<unsigned char, cdiv(rows *cols, 8)> &bit_matrix) {
  size_t outer_idx = 0;
  for (unsigned char val : bit_matrix) {
    for (int i = 0; i < 8; i++) {
      int idx = outer_idx * 8 + i;
      if (idx < rows * cols) {
        matrix[idx] = static_cast<int>((val >> i) & 0b1);
      }
    }
    outer_idx++;
  }
}

template <size_t M, size_t K, size_t N, size_t G>
void initialize_data(std::array<int, M * K> &W_r_unpacked,
                     std::array<float, G> &z, std::array<float, G> &s,
                     std::array<float, K * N> &x,
                     std::array<float, M * N> &out) {}

template <size_t M, size_t K, size_t N, size_t G>
void populate_values(std::array<int, M * K> &W_unpacked,
                     std::array<float, G> &z, std::array<float, G> &s,
                     std::array<float, K * N> &x,
                     std::array<float, M * N> &out) {
  std::mt19937 gen(42);
  randint<int, M * K>(W_unpacked, gen, 0, 1);
  std::fill_n(begin(z), G, 0.0);
  std::fill_n(begin(s), G, 1.0);
  randint<float, K * N>(x, gen, -1, 4);
  std::fill_n(begin(out), M * N, 0.0);
}

template <size_t M, size_t K, size_t N, size_t GS,
          void (*Kernel)(const unsigned char *, const float *, const float *,
                         const float *, float *, int, int, int, int)>
void launch(const std::array<unsigned char, cdiv(M * K, 8)> &w,
            const std::array<float, cdiv(M * K, GS)> &z,
            const std::array<float, cdiv(M * K, GS)> &s,
            const std::array<float, K * N> &x, std::array<float, M * N> &out) {
  unsigned char *w_d;
  float *z_d, *s_d, *x_d, *out_d;
  cudaMalloc(&w_d, sizeof(w));
  cudaMalloc(&z_d, sizeof(z));
  cudaMalloc(&s_d, sizeof(s));
  cudaMalloc(&x_d, sizeof(x));
  cudaMalloc(&out_d, sizeof(out));
  cudaMemcpy(w_d, w.data(), sizeof(w), cudaMemcpyHostToDevice);
  cudaMemcpy(z_d, z.data(), sizeof(z), cudaMemcpyHostToDevice);
  cudaMemcpy(s_d, s.data(), sizeof(s), cudaMemcpyHostToDevice);
  cudaMemcpy(x_d, x.data(), sizeof(x), cudaMemcpyHostToDevice);
  check_cuda_errors(__LINE__);

  static constexpr int kTileWidth = 32;
  dim3 threadsPerBlock(kTileWidth, kTileWidth, 1);
  dim3 blocksPerGrid(cdiv(N, kTileWidth), cdiv(M, kTileWidth), 1);
  spdlog::info("threadsPerBlock: ({}, {}, {})", threadsPerBlock.x,
               threadsPerBlock.y, threadsPerBlock.z);
  spdlog::info("blocksPerGrid: ({}, {}, {})", blocksPerGrid.x, blocksPerGrid.y,
               blocksPerGrid.z);

  Kernel<<<blocksPerGrid, threadsPerBlock>>>(w_d, z_d, s_d, x_d, out_d, M, K, N,
                                             GS);

  cudaDeviceSynchronize(); // block on kernel completion
  check_cuda_errors(__LINE__);
  cudaMemcpy(out.data(), out_d, sizeof(out), cudaMemcpyDeviceToHost);

  cudaFree(w_d);
  cudaFree(z_d);
  cudaFree(s_d);
  cudaFree(x_d);
  cudaFree(out_d);
  check_cuda_errors(__LINE__);
}

int main() {
  auto logger = spdlog::default_logger();
  logger->set_level(spdlog::level::info);
  logger->set_pattern("[%^%l%$] %v");
  spdlog::info(show_runtime_info(0));

  static constexpr size_t M = 8;
  static constexpr size_t K = 256;
  static constexpr size_t N = 8;
  static constexpr size_t GS = 64;
  static constexpr size_t G = cdiv(M * K, /* group size = */ GS);

  std::array<int, M * K> w_orig;               // w w/o bit packing
  std::array<unsigned char, cdiv(M * K, 8)> w; // w w/ bit packing
  std::array<float, G> z;
  std::array<float, G> s;
  std::array<float, K * N> x;
  std::array<float, M * N> out;
  populate_values<M, K, N, G>(w_orig, z, s, x, out);
  bit_encoder<M, K>(w_orig, w);

  spdlog::info(show<int, M, K>(w_orig, "w_orig"));
  spdlog::info(show<float, K, N>(x, "x"));
  spdlog::info(show<float, 1, G>(z, "z"));
  spdlog::info(show<float, 1, G>(s, "s"));

  launch<M, K, N, GS, mm1b>(w, z, s, x, out);

  spdlog::info(show<float, M, N>(out, "out"));

  cudaDeviceReset();
  spdlog::info("Done");
}
