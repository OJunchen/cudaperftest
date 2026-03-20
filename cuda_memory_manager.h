/*
 * cuda_memory_manager.h - CUDA memory management for cudaperftest
 */

#ifndef CUDA_MEMORY_MANAGER_H_
#define CUDA_MEMORY_MANAGER_H_

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>

// Base memory class
class BaseMemory {
public:
    virtual ~BaseMemory() = default;
    virtual void* getBuffer() const = 0;
    virtual size_t getBufferSize() const = 0;
    virtual int getDeviceId() const = 0;
};

// Host memory implementation
class HostMemory : public BaseMemory {
private:
    void* buffer;
    size_t size;
    int deviceId;
    bool pinned;

public:
    HostMemory(size_t size, int deviceId, bool pinned);
    ~HostMemory();
    
    void* getBuffer() const override { return buffer; }
    size_t getBufferSize() const override { return size; }
    int getDeviceId() const override { return deviceId; }
    bool isPinned() const { return pinned; }
};

// Device memory implementation
class DeviceMemory : public BaseMemory {
private:
    void* buffer;
    size_t size;
    int deviceId;

public:
    DeviceMemory(size_t size, int deviceId);
    ~DeviceMemory();
    
    void* getBuffer() const override { return buffer; }
    size_t getBufferSize() const override { return size; }
    int getDeviceId() const override { return deviceId; }
};

// Memory factory
class MemoryFactory {
public:
    static std::unique_ptr<HostMemory> createHostMemory(size_t size, int deviceId, bool pinned);
    static std::unique_ptr<DeviceMemory> createDeviceMemory(size_t size, int deviceId);
};

// CUDA context manager
class CudaContext {
private:
    int deviceId;
    cudaStream_t stream;
    cudaStream_t streamForward;
    cudaStream_t streamReverse;
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    cudaEvent_t forwardStartEvent;
    cudaEvent_t forwardStopEvent;
    cudaEvent_t reverseStartEvent;
    cudaEvent_t reverseStopEvent;
    bool bidirectional;

public:
    explicit CudaContext(int deviceId, bool bidir = false);
    ~CudaContext();
    
    void synchronize();
    void synchronizeAllStreams();
    void recordStart();
    void recordStop();
    void recordBidirectionalStart();
    void recordBidirectionalStop();
    float getElapsedTime();  // Returns time in ms
    float getForwardElapsedTime();
    float getReverseElapsedTime();
    float getBidirectionalElapsedTime();  // max of forward and reverse
    cudaStream_t getStream() const { return stream; }
    cudaStream_t getStreamForward() const { return streamForward; }
    cudaStream_t getStreamReverse() const { return streamReverse; }
    int getDeviceId() const { return deviceId; }
    bool isBidirectional() const { return bidirectional; }
};

// Utility macros
#define CUDA_ASSERT(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                                    ": " + cudaGetErrorString(err) + " (" + #call + ")"); \
        } \
    } while(0)

#endif  // CUDA_MEMORY_MANAGER_H_
