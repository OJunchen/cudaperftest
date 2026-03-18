cmake_minimum_required(VERSION 3.10)
project(cudaperftest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Source files
set(SOURCES
    main.cpp
    test_runner.cpp
    cuda_memory_manager.cpp
)

# Header files
set(HEADERS
    test_config.h
    test_runner.h
    cuda_memory_manager.h
)

# Create executable
add_executable(cudaperftest ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(cudaperftest PRIVATE ${CUDA_INCLUDE_DIRS})

# Link CUDA libraries
target_link_libraries(cudaperftest ${CUDA_LIBRARIES} cuda)

# Set CUDA architectures (adjust based on your GPU)
set_target_properties(cudaperftest PROPERTIES
    CUDA_ARCHITECTURES "70;80"
)

# Compiler flags for release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
