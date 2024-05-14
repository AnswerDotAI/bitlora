// Pure Cpp implementation of linear.fwd and linear.bwd

#include <torch/extension.h>
#include <vector>

torch::Tensor forward(torch::Tensor weight, torch::Tensor x) {
    return torch::mm(x, weight.t());
}

std::vector<torch::Tensor> backward(torch::Tensor d, torch::Tensor weight, torch::Tensor x) {
    torch::Tensor dx = torch::mm(d, weight);
    torch::Tensor dWeight = torch::mm(d.t(), x);
    return {dWeight, dx};
}

// "custom_lin_cpp" is the name param in torch.utils.cpp_extension.load
PYBIND11_MODULE(custom_lin_cpp, m) {
    m.def("forward", &forward, "Custom linear forward");
    m.def("backward", &backward, "Custom linear backward");
}
