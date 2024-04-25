cmake_minimum_required(VERSION 3.11)

include(FetchContent)

project(run_bitlora LANGUAGES CXX CUDA)

# Force C++17

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# if uncommented - create compile_commands.json logging all compile commands
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON) 

# Load dependencies - nanobench for benchmarking, spdlog for logging &
# debugging

FetchContent_Declare(cutlass 
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git 
  GIT_TAG 5c447dd84f8ae0e1d48ff9a2eae26ce8c4958101
)
FetchContent_MakeAvailable(cutlass)

FetchContent_Declare(nanobench
  GIT_REPOSITORY https://github.com/martinus/nanobench.git
  GIT_TAG 9f871ad520e040078597f4194080cc073a765061
)
FetchContent_MakeAvailable(nanobench)

FetchContent_Declare(spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git 
  GIT_TAG a2b4262090fd3f005c2315dcb5be2f0f1774a005
)
FetchContent_MakeAvailable(spdlog)

# Turn off optimizations for fast builds
option(FASTBUILD "Option to enable fast builds" OFF)

if(FASTBUILD)
  set(CMAKE_BUILD_TYPE None)  # Avoid default flags of predefined build types
  set(CMAKE_CUDA_FLAGS "-O0" CACHE STRING "Optimization flags" FORCE)
endif()

add_executable(run_bitlora run_bitlora.cu)
target_compile_features(run_bitlora PUBLIC cxx_std_17)
set_target_properties(run_bitlora PROPERTIES CUDA_STANDARD 17)
set_target_properties(run_bitlora PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(run_bitlora PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(run_bitlora nanobench spdlog::spdlog CUTLASS)
target_include_directories(run_bitlora PRIVATE ${CUTLASS_INCLUDE_DIR})
target_include_directories(run_bitlora PRIVATE ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR})
set_target_properties(run_bitlora PROPERTIES CUDA_ARCHITECTURES "86")
