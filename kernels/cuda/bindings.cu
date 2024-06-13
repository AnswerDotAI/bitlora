#include <torch/extension.h>
#include <cstdint> // uint32_t

/*
   HQQ 3 bit + Dora as a single kernel, no tiling

   x is stored row major (M x K)
   wq/w, loraA, and loraB are row major (K x N)

  Convention matches pytorch

  wq : (K, N / 10) K = input size, N = output size
  w : (K, N) K = input size, N = output size
  z : (K x N / GS) K = input size, N = output size, GS = group size
  s : (K x N / GS) K = input size, N = output size, GS = group size
  x : (M x K) M = batch size, K = input size
  loraA : (K x R) K = input size, R = rank
  loraB : (R x N) R = rank, N = output size



  Avoids materializing x @ loraA and simply calculates the output in a single
  pass.

  loraOut[i, j] = sum_{r=0}^{R-1} sum_{k=0}^{K-1} x[i, k] * loraA[k, r] *
  loraB[r, j]


*/
__global__ void qdorav1(const uint32_t *wq, float *w, const float *z,
                        const float *s, const float *x, float *out, int M,
                        int K, int N, int GS, const float *loraA,
                        const float *loraB, float *loraOut, size_t R,
                        float* doraScale) {

  static constexpr size_t kBitsPerVal = 10;

  /* dequant wq */
  const size_t threadId = (blockIdx.x * blockDim.x + threadIdx.x);
  const size_t wRow = threadId / K;
  const size_t wCol = threadId % K;

  // dequant wq -> w
  const size_t outer_index = (wRow * K + wCol) / kBitsPerVal;
  const size_t inner_index = (wRow * K + wCol) % kBitsPerVal;
  const size_t shift_amount = 32 - 2 - 3 - inner_index * 3;
  const size_t group_id = (wRow * K + wCol) / GS;
  w[wRow * K + wCol] =
      static_cast<float>(wq[outer_index] >> shift_amount & 0b111) *
          s[group_id] -
      z[group_id];

  __syncthreads();

  // naive 3 matmuls: base layer, loraA x loraB
  const size_t outRow = threadId / N;
  const size_t outCol = threadId % N;
  for (size_t idx = 0; idx < R * K; ++idx) {
    if (outRow < M && outCol < N && idx < K) {
      out[outRow * N + outCol] += x[outRow * K + idx] * w[idx * N + outCol];
    } else {
      const size_t r = idx / K;
      const size_t k = idx % K;
      loraOut[outRow * N + outCol] +=
          x[outRow * K + k] * loraA[k * R + r] * loraB[r * N + outCol];
    }
  }

  __syncthreads();

  out[outRow * N + outCol] += loraOut[outRow * N + outCol] * doraScale[outCol];
}

void qdora(torch::Tensor wq, torch::Tensor z, torch::Tensor s, torch::Tensor x,
           torch::Tensor out, int M, int K, int N, int GS, torch::Tensor loraA,
           torch::Tensor loraB, size_t R, torch::Tensor doraScale) {
  int blockSize = 256;
  int numBlocks = (M * N + blockSize - 1) / blockSize;
  float *wDev;       // buffer for unquantized wq
  float *loraOutDev; // buffer for lora output (prior to adding to base layer
                     // output)
  cudaMalloc(&wDev, sizeof(float) * M * K);
  cudaMalloc(&loraOutDev, sizeof(float) * M * N);
  qdorav1<<<numBlocks, blockSize>>>(
      reinterpret_cast<uint32_t*>(wq.data_ptr<int32_t>()), wDev, z.data_ptr<float>(), s.data_ptr<float>(),
      x.data_ptr<float>(), out.data_ptr<float>(), M, K, N, GS,
      loraA.data_ptr<float>(), loraB.data_ptr<float>(), loraOutDev, R,
      doraScale.data_ptr<float>());
  cudaFree(wDev);
  cudaFree(loraOutDev);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("qdora", &qdora, "QDORA"); }
