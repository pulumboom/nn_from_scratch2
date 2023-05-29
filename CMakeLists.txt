cmake_minimum_required(VERSION 3.23)
project(nn_from_scratch)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src/layers)
include_directories(src/activation_functions)
include_directories(src/criterions)
include_directories(src/base)
include_directories(src/optimizers)
include_directories(src/utils)
include_directories(lib/eigen)
include_directories(lib/EigenRand)

add_executable(nn_from_scratch
        src/layers/LinearLayer.cpp
        src/layers/Sequential.cpp
        src/activation_functions/ReLU.cpp
        src/activation_functions/Sigmoid.cpp
        src/activation_functions/Softmax.cpp
        src/criterions/MSE.cpp
        src/criterions/CrossEntropyLoss.cpp
        src/optimizers/SGD.cpp
        src/utils/DataProcessing.cpp
        src/main.cpp)
