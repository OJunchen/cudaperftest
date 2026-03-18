/*
 * cuda_memory_manager.cpp - CUDA memory management implementation
 */

#include "cuda_memory_manager.h"
#include <cstring>

// HostMemory implementation
HostMemory::HostMemory(size_t size, int deviceId, bool pinned)
    : buffer(nullptr), size(size), deviceId(deviceId), pinned(pinned) {
    if (pinned) {
        CUDA_ASSERT(cudaHostAlloc(&buffer, size, cudaHostAllocDefault));
    } else {
        buffer = malloc(size);
        if (!buffer) throw std::bad_alloc();
    }
}

HostMemory::~HostMemory() {
    if (buffer) {
        if (pinned) {
            cudaFreeHost(buffer);
        } else {
            free(buffer);
        }
    }
}

// DeviceMemory implementation
DeviceMemory::DeviceMemory(size_t size, int deviceId)
    : buffer(nullptr), size(size), deviceId(deviceId) {
    CUDA_ASSERT(cudaSetDevice(deviceId));
    CUDA_ASSERT(cudaMalloc(&buffer, size));
}

DeviceMemory::~DeviceMemory() {
    if (buffer) {
        cudaSetDevice(deviceId);
        cudaFree(buffer);
    }
}

// MemoryFactory implementation
std::unique_ptr<HostMemory> MemoryFactory::createHostMemory(size_t size, int deviceId, bool pinned) {
    return std::make_unique<HostMemory>(size, deviceId, pinned);
}

std::unique_ptr<DeviceMemory> MemoryFactory::createDeviceMemory(size_t size, int deviceId) {
    return std::make_unique<DeviceMemory>(size, deviceId);
}

// CudaContext implementation
CudaContext::CudaContext(int deviceId)
    : deviceId(deviceId), stream(nullptr) {
    CUDA_ASSERT(cudaSetDevice(deviceId));
    CUDA_ASSERT(cudaStreamCreate(&stream));
    CUDA_ASSERT(cudaEventCreate(&startEvent));
    CUDA_ASSERT(cudaEventCreate(&stopEvent));
}

CudaContext::~CudaContext() {
    if (stream) cudaStreamDestroy(stream);
    if (startEvent) cudaEventDestroy(startEvent);
    if (stopEvent) cudaEventDestroy(stopEvent);
}

void CudaContext::synchronize() {
    CUDA_ASSERT(cudaStreamSynchronize(stream));
}

void CudaContext::recordStart() {
    CUDA_ASSERT(cudaEventRecord(startEvent, stream));
}

void CudaContext::recordStop() {
    CUDA_ASSERT(cudaEventRecord(stopEvent, stream));
}

float CudaContext::getElapsedTime() {
    float elapsedTime;
    CUDA_ASSERT(cudaEventSynchronize(stopEvent));
    CUDA_ASSERT(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));
    return elapsedTime;
}
