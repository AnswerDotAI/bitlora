#ifndef KERNEL_TOOLS_CUH
#define KERNEL_TOOLS_CUH

#include <stdexcept>
#include <string>
#include <utility>

// Convenience functions for cuda runtime

std::string show_runtime_info(int device_id) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  std::string output;
  output += "\n\nDevice Name: " + std::string(prop.name) + "\n";
  output += "Compute Capability: " + std::to_string(prop.major) + "." +
            std::to_string(prop.minor) + "\n";
  output += "Global Memory: " + std::to_string(prop.totalGlobalMem) + "\n";
  output +=
      "Multiprocessor Count: " + std::to_string(prop.multiProcessorCount) +
      "\n";
  output += "Warp Size: " + std::to_string(prop.warpSize) + "\n";
  output +=
      "Max Threads Per Block: " + std::to_string(prop.maxThreadsPerBlock) +
      "\n";
  output += "Max Threads Dim: " + std::to_string(prop.maxThreadsDim[0]) + " " +
            std::to_string(prop.maxThreadsDim[1]) + " " +
            std::to_string(prop.maxThreadsDim[2]) + "\n";
  output += "Max Grid Size: " + std::to_string(prop.maxGridSize[0]) + " " +
            std::to_string(prop.maxGridSize[1]) + " " +
            std::to_string(prop.maxGridSize[2]) + "\n";
  output +=
      "Max Shared Memory Per Block: " + std::to_string(prop.sharedMemPerBlock) +
      "\n";
  output +=
      "Max Registers Per Block: " + std::to_string(prop.regsPerBlock) + "\n";
  output += "Max Threads Per Multiprocessor: " +
            std::to_string(prop.maxThreadsPerMultiProcessor) + "\n";
  output += "Max Shared Memory Per Multiprocessor: " +
            std::to_string(prop.sharedMemPerMultiprocessor) + "\n";
  output += "Max Registers Per Multiprocessor: " +
            std::to_string(prop.regsPerMultiprocessor) + "\n";
  output += "Max Threads Per Warp: " +
            std::to_string(prop.maxThreadsPerMultiProcessor /
                           prop.multiProcessorCount) +
            "\n\n";
  return output;
}

// return pair with first element being error no error and right being message
void check_cuda_errors(int line) {
  cudaError_t err = cudaGetLastError();
  std::string output;
  if (err != cudaSuccess) {
    output = "Line " + std::to_string(line) + " : CUDA error: " +
             cudaGetErrorString(err) + "\n\n";
    printf("cuda error: %s\n", output.c_str());
  }
}

// Dimension handling

constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// Populating array values

template <typename numtype, size_t size>
void randint(std::array<numtype, size> &a, std::mt19937 &gen, int min,
               int max) {
  std::uniform_int_distribution<> dist(min, max);
  for (int i = 0; i < size; i++) {
    a[i] = static_cast<numtype>(dist(gen));
  }
}

// Visualization / observability


template <typename numtype, size_t rows, size_t cols>
std::string show(std::array<numtype, rows * cols> a, std::string name = "") {
  std::string output = "\n";
  if (name != "") {
    output += name + " : \n";
  }
  // spacing as log10 of max value
  int spacing = 1;
  numtype max = *std::max_element(a.begin(), a.end());
  if constexpr (std::is_same<numtype, int>::value) {
    spacing = std::max(0, (int)log10(max + .01)) + 2;
  } else if constexpr (std::is_same<numtype, float>::value) {
    spacing = std::max(0, (int)log10(max + .01)) + 5;
  } else {
    throw std::runtime_error("Unsupported number type for show()");
  }

  // print to stdout line break for each row
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      char buffer[50];
      if constexpr (std::is_same<numtype, int>::value) {
        sprintf(buffer, "%*d", spacing, a[i * cols + j]);
      } else if constexpr (std::is_same<numtype, float>::value) {
        sprintf(buffer, "%*.*f", spacing, 1, a[i * cols + j]);
      } else {
        throw std::runtime_error("Unsupported number type for show()");
      }
      output += buffer;
    }
    output += "\n";
  }
  output += "\n";
  return output;
}


#endif // KERNEL_TOOLS_CUH
