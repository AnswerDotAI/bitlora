#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

// //  kernels (suuper naive version)
// forward
template <typename t>
__global__ void cuda_forward_kernel(
    /* inputs */ const t* __restrict__ weight, const t* __restrict__ x,
    /* output */ t* __restrict__ out,
    /* sizes  */ size_t m, size_t n
) {
  const int mid = blockIdx.x * blockDim.x + threadIdx.x;
  const int bid = blockIdx.y;
  
  if (mid >= m) { return; }

  printf("in kernel: blockidx=(%d,%d), blockdim.x=%d, threadidx.x=%d | m=%lu, n=%lu | mid=%d, bid=%d | bid*m+mid=%lu\n", blockIdx.x, blockIdx.y, blockDim.x, threadIdx.x, m,n, mid, bid, bid*m+mid);

  t acc = 0.0;
  for (int i = 0; i<n; i++) {
    printf("loop: threadidx.x=%d, mid*n+i=%lu, bid*n+i=%lu\n", threadIdx.x, mid*n+i, bid*n+i);

    acc += weight[mid*n+i] * x[bid*n+i];
  }
 out[bid*m+mid] = acc;
}

// backward
template <typename t>
__global__ void cuda_backward_kernel(
    /* inputs */ const t* __restrict__ d, const t* __restrict__ weight, const t* __restrict__ x,
    /* output */ t* __restrict__ d_weight, t* __restrict__ d_x,
    /* sizes  */ size_t batch_size, size_t m, size_t n
) {
  const int nid = blockIdx.x * blockDim.x + threadIdx.x;
  const int bid = blockIdx.y; // batch
  const int mid = blockIdx.z;

  if (nid >= n) { return; }

  // Compute d_x[bid, nid] by iterating over m
  t acc = 0.0;
  for (int i = 0; i < m; i++) { acc += d[bid*m+i] * weight[i*n+nid]; }
  d_x[bid*n+nid] = acc;

  // Compute d_weight[mid, nid] by iterating over b
  acc = 0.0;
  for (int i = 0; i < batch_size; i++) { acc += d[i*m+mid] * x[i*n+nid]; }
  d_weight[mid*n+nid] = acc;
}

// // kernel launcher
// forward
torch::Tensor cuda_forward(torch::Tensor weight, torch::Tensor x) {
  // shapes: (batch_size, n) @ (m,n).t -> (batch-size, m)
  const auto m = weight.size(0);
  const auto n = weight.size(1);
  const auto batch_size = x.size(0);

  auto out = torch::zeros({batch_size, m});

  const int threads = 1024;
  const dim3 blocks(cdiv(m,threads), batch_size);

  std::cout << "--- started cuda_forward" << std::endl;
  std::cout << "Shape of weight: " << weight.sizes() << std::endl;
  std::cout << "Shape of x: " << x.sizes() << std::endl;
  std::cout << "Type of m: '" << typeid(m).name() << "', Value: " << m << std::endl;
  std::cout << "Type of n: '" << typeid(n).name() << "', Value: " << n << std::endl;
  std::cout << "Type of batch_size: '" << typeid(batch_size).name() << "', Value: " << batch_size << std::endl;
  std::cout << "Shape of out: [" << batch_size << ", " << m << "]" << std::endl;
  std::cout << "Threads per block: " << threads << std::endl;
  std::cout << "Blocks: [" << blocks.x << ", " << blocks.y << "]" << std::endl;

  AT_DISPATCH_FLOATING_TYPES(x.type(), "cuda_forward", ([&] {
    cuda_forward_kernel<scalar_t><<<blocks, 1>>>( //<<<blocks, threads>>>
        /* inputs */ weight.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
        /* output */ out.data_ptr<scalar_t>(),
        /* sizes  */ m, n
    );
  }));

  std::cout << "- kernel launched" << std::endl;

  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
      std::cerr << "CUDA kernel error: " << cudaGetErrorString(error) << std::endl;
  } else {
      std::cout << "CUDA kernel completed successfully." << std::endl;
  }

  std::cout << "- kernel completed" << std::endl;

  return out;
}

// backward
std::vector<torch::Tensor> cuda_backward(torch::Tensor d, torch::Tensor weight, torch::Tensor x) {
  const auto m = weight.size(0);
  const auto n = weight.size(1);
  const auto batch_size = x.size(0);

  auto d_x = torch::zeros_like(x);
  auto d_weight = torch::zeros_like(weight);

  const int threads = 1024;
  const dim3 blocks(cdiv(n, threads), batch_size, m);

  AT_DISPATCH_FLOATING_TYPES(d.type(), "cuda_backward", ([&] {
    cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        /* inputs */ d.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
        /* output */ d_weight.data_ptr<scalar_t>(), d_x.data_ptr<scalar_t>(),
        /* sizes  */ batch_size, m, n
    );
  }));

  return {d_weight, d_x};
}
