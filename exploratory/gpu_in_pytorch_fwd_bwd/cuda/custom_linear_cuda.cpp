// Cpp interface of Cuda-C implementation of linear.fwd and linear.bwd

#include <torch/extension.h>
#include <vector>
#include <iostream>

// CUDA forward declarations
std::vector<torch::Tensor> cuda_forward(torch::Tensor weight, torch::Tensor x);
std::vector<torch::Tensor> cuda_backward(torch::Tensor d, torch::Tensor weight, torch::Tensor x);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> forward(torch::Tensor weight, torch::Tensor x) {
  std::cout << "--- started forward" << std::endl;
  CHECK_INPUT(weight);
  CHECK_INPUT(x);
  std::cout << "Shape of weight: " << weight.sizes() << std::endl;
  std::cout << "Shape of x: " << x.sizes() << std::endl;
  return cuda_forward(weight, x);
}

std::vector<torch::Tensor> backward(torch::Tensor d, torch::Tensor weight, torch::Tensor x) {
  CHECK_INPUT(d);
  CHECK_INPUT(weight);
  CHECK_INPUT(x);
  return cuda_backward(d, weight, x);
}

// "custom_lin_cpp" is the name param in torch.utils.cpp_extension.load
PYBIND11_MODULE(custom_lin_cuda, m) {
    m.def("forward", &forward, "Custom linear forward");
    m.def("backward", &backward, "Custom linear backward");
}
