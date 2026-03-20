/*
 * latency_test.cpp - Latency test implementation for cudaperftest
 */

#include "latency_test.h"
#include <iostream>
#include <iomanip>
#include <chrono>

LatencyTestRunner::LatencyTestRunner(const TestConfig& cfg, const TestEnvironment& env)
    : TestRunner(cfg, env) {
    testName = "Latency Test";
}

void LatencyTestRunner::run() {
    std::vector<int> devices = env.useAllDevices ? std::vector<int>{0} : env.targetDevices;
    
    std::vector<std::string> sizeLabels;
    std::vector<std::vector<double>> meanMatrix;
    std::vector<std::vector<double>> p99Matrix;
    
    std::vector<unsigned long long> sizes = {
        1, 4, 16, 64, 256, 1024, 4096, 16384, 65536,  // 1B to 64KB
        262144, 1048576, 4194304, 16777216, 33554432   // 256KB to 32MB
    };
    
    for (auto size : sizes) {
        sizeLabels.push_back(TestUtils::formatSize(size));
    }
    
    for (int deviceId : devices) {
        if (!testInitialize(deviceId)) continue;
        
        std::vector<double> deviceMean;
        std::vector<double> deviceP99;
        
        for (auto size : sizes) {
            bool memoryOk;
            if (config.bidirectional) {
                memoryOk = bidirectionalMemoryApply(size, deviceId);
            } else {
                memoryOk = memoryApply(size, deviceId);
            }
            
            if (!memoryOk) continue;
            
            try {
                stats.reset();
                if (config.bidirectional) {
                    doBidirectionalMemcpy(size, TestSizeConfig::LATENCY_ITERATIONS);
                } else {
                    doMemcpy(size, TestSizeConfig::LATENCY_ITERATIONS);
                }
                
                double mean, p99;
                calculateLatency(stats, mean, p99);
                deviceMean.push_back(mean);
                deviceP99.push_back(p99);
            } catch (const std::exception& e) {
                std::cerr << "Test failed for size " << size << ": " << e.what() << std::endl;
            }
            
            srcMemory.reset();
            dstMemory.reset();
            bidirSrcMemory.reset();
            bidirDstMemory.reset();
        }
        
        cleanup();
        meanMatrix.push_back(deviceMean);
        p99Matrix.push_back(deviceP99);
    }
    
    TestUtils::printLatencyMatrix(meanMatrix, p99Matrix, sizeLabels, testName);
}

bool LatencyTestRunner::doMemcpy(unsigned long long size, int iterations) {
    void* src = srcMemory->getBuffer();
    void* dst = dstMemory->getBuffer();
    
    // Warmup - use synchronous transfers
    for (unsigned int w = 0; w < env.warmupIterations; w++) {
        switch (config.direction) {
            case TransferDirection::H2D:
                CUDA_ASSERT(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
                break;
            case TransferDirection::D2H:
                CUDA_ASSERT(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
                break;
            case TransferDirection::D2D:
                CUDA_ASSERT(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
                break;
        }
    }
    
    // Measurement - use CPU high-precision timer for end-to-end latency
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        switch (config.direction) {
            case TransferDirection::H2D:
                CUDA_ASSERT(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
                break;
            case TransferDirection::D2H:
                CUDA_ASSERT(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
                break;
            case TransferDirection::D2D:
                CUDA_ASSERT(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
                break;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        
        double latencyUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        stats.record(latencyUs);
    }
    
    return true;
}

bool LatencyTestRunner::doBidirectionalMemcpy(unsigned long long size, int iterations) {
    // For latency tests, bidirectional means round-trip:
    // We execute forward memcpy, then reverse memcpy (sequential, not concurrent)
    // This measures the round-trip latency
    
    void* src = srcMemory->getBuffer();
    void* dst = dstMemory->getBuffer();
    void* bidirSrc = bidirSrcMemory->getBuffer();
    void* bidirDst = bidirDstMemory->getBuffer();
    
    // Warmup - use synchronous transfers
    for (unsigned int w = 0; w < env.warmupIterations; w++) {
        // Forward transfer
        switch (config.direction) {
            case TransferDirection::H2D:
                CUDA_ASSERT(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
                break;
            case TransferDirection::D2H:
                CUDA_ASSERT(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
                break;
            case TransferDirection::D2D:
                CUDA_ASSERT(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
                break;
        }
        
        // Reverse transfer (round-trip)
        switch (config.direction) {
            case TransferDirection::H2D:
                CUDA_ASSERT(cudaMemcpy(bidirDst, bidirSrc, size, cudaMemcpyDeviceToHost));
                break;
            case TransferDirection::D2H:
                CUDA_ASSERT(cudaMemcpy(bidirDst, bidirSrc, size, cudaMemcpyHostToDevice));
                break;
            case TransferDirection::D2D:
                CUDA_ASSERT(cudaMemcpy(bidirDst, bidirSrc, size, cudaMemcpyDeviceToDevice));
                break;
        }
    }
    
    // Measurement - use CPU high-precision timer for end-to-end round-trip latency
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Forward transfer
        switch (config.direction) {
            case TransferDirection::H2D:
                CUDA_ASSERT(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
                break;
            case TransferDirection::D2H:
                CUDA_ASSERT(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
                break;
            case TransferDirection::D2D:
                CUDA_ASSERT(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
                break;
        }
        
        // Reverse transfer (round-trip)
        switch (config.direction) {
            case TransferDirection::H2D:
                CUDA_ASSERT(cudaMemcpy(bidirDst, bidirSrc, size, cudaMemcpyDeviceToHost));
                break;
            case TransferDirection::D2H:
                CUDA_ASSERT(cudaMemcpy(bidirDst, bidirSrc, size, cudaMemcpyHostToDevice));
                break;
            case TransferDirection::D2D:
                CUDA_ASSERT(cudaMemcpy(bidirDst, bidirSrc, size, cudaMemcpyDeviceToDevice));
                break;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        
        double latencyUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        stats.record(latencyUs);
    }
    
    return true;
}
