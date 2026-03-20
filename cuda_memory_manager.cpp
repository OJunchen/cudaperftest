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
        CUDA_ASSERT(cudaSetDevice(deviceId));
        CUDA_ASSERT(cudaFree(buffer));
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
CudaContext::CudaContext(int deviceId, bool bidir)
    : deviceId(deviceId), stream(nullptr), streamForward(nullptr), 
      streamReverse(nullptr), bidirectional(bidir) {
    CUDA_ASSERT(cudaSetDevice(deviceId));
    CUDA_ASSERT(cudaStreamCreate(&stream));
    CUDA_ASSERT(cudaEventCreate(&startEvent));
    CUDA_ASSERT(cudaEventCreate(&stopEvent));
    
    if (bidirectional) {
        CUDA_ASSERT(cudaStreamCreate(&streamForward));
        CUDA_ASSERT(cudaStreamCreate(&streamReverse));
        CUDA_ASSERT(cudaEventCreate(&forwardStartEvent));
        CUDA_ASSERT(cudaEventCreate(&forwardStopEvent));
        CUDA_ASSERT(cudaEventCreate(&reverseStartEvent));
        CUDA_ASSERT(cudaEventCreate(&reverseStopEvent));
    }
}

CudaContext::~CudaContext() {
    if (stream) cudaStreamDestroy(stream);
    if (streamForward) cudaStreamDestroy(streamForward);
    if (streamReverse) cudaStreamDestroy(streamReverse);
    if (startEvent) cudaEventDestroy(startEvent);
    if (stopEvent) cudaEventDestroy(stopEvent);
    if (forwardStartEvent) cudaEventDestroy(forwardStartEvent);
    if (forwardStopEvent) cudaEventDestroy(forwardStopEvent);
    if (reverseStartEvent) cudaEventDestroy(reverseStartEvent);
    if (reverseStopEvent) cudaEventDestroy(reverseStopEvent);
}

void CudaContext::synchronize() {
    CUDA_ASSERT(cudaStreamSynchronize(stream));
}

void CudaContext::synchronizeAllStreams() {
    CUDA_ASSERT(cudaStreamSynchronize(stream));
    if (bidirectional) {
        CUDA_ASSERT(cudaStreamSynchronize(streamForward));
        CUDA_ASSERT(cudaStreamSynchronize(streamReverse));
    }
}

void CudaContext::recordStart() {
    CUDA_ASSERT(cudaEventRecord(startEvent, stream));
}

void CudaContext::recordStop() {
    CUDA_ASSERT(cudaEventRecord(stopEvent, stream));
}

void CudaContext::recordBidirectionalStart() {
    if (bidirectional) {
        CUDA_ASSERT(cudaEventRecord(forwardStartEvent, streamForward));
        CUDA_ASSERT(cudaEventRecord(reverseStartEvent, streamReverse));
    }
}

void CudaContext::recordBidirectionalStop() {
    if (bidirectional) {
        CUDA_ASSERT(cudaEventRecord(forwardStopEvent, streamForward));
        CUDA_ASSERT(cudaEventRecord(reverseStopEvent, streamReverse));
    }
}

float CudaContext::getElapsedTime() {
    float elapsedTime;
    CUDA_ASSERT(cudaEventSynchronize(stopEvent));
    CUDA_ASSERT(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));
    return elapsedTime;
}

float CudaContext::getForwardElapsedTime() {
    float elapsedTime;
    CUDA_ASSERT(cudaEventSynchronize(forwardStopEvent));
    CUDA_ASSERT(cudaEventElapsedTime(&elapsedTime, forwardStartEvent, forwardStopEvent));
    return elapsedTime;
}

float CudaContext::getReverseElapsedTime() {
    float elapsedTime;
    CUDA_ASSERT(cudaEventSynchronize(reverseStopEvent));
    CUDA_ASSERT(cudaEventElapsedTime(&elapsedTime, reverseStartEvent, reverseStopEvent));
    return elapsedTime;
}

float CudaContext::getBidirectionalElapsedTime() {
    // For bidirectional bandwidth, we need the time when BOTH directions complete
    float forwardMs = getForwardElapsedTime();
    float reverseMs = getReverseElapsedTime();
    return std::max(forwardMs, reverseMs);
}
