/*
 * latency_test.cpp - Latency test implementation for cudaperftest
 */

#include "latency_test.h"
#include <iostream>
#include <iomanip>

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
        1, 4, 16, 64, 256, 1024, 4096, 16384, 65536  // 1B to 64KB
    };
    
    for (auto size : sizes) {
        sizeLabels.push_back(TestUtils::formatSize(size));
    }
    
    for (int deviceId : devices) {
        if (!testInitialize(deviceId)) continue;
        
        std::vector<double> deviceMean;
        std::vector<double> deviceP99;
        
        for (auto size : sizes) {
            if (!memoryApply(size, deviceId)) continue;
            
            stats.reset();
            doMemcpy(size, TestSizeConfig::LATENCY_ITERATIONS);
            
            double mean, p99;
            calculateLatency(stats, mean, p99);
            deviceMean.push_back(mean);
            deviceP99.push_back(p99);
            
            cleanup();
        }
        
        meanMatrix.push_back(deviceMean);
        p99Matrix.push_back(deviceP99);
    }
    
    TestUtils::printLatencyMatrix(meanMatrix, p99Matrix, sizeLabels, testName);
}

bool LatencyTestRunner::doMemcpy(unsigned long long size, int iterations) {
    void* src = srcMemory->getBuffer();
    void* dst = dstMemory->getBuffer();
    cudaStream_t stream = cudaContext->getStream();
    
    // Warmup
    for (unsigned int w = 0; w < env.warmupIterations; w++) {
        switch (config.direction) {
            case TransferDirection::H2D:
                CUDA_ASSERT(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
                break;
            case TransferDirection::D2H:
                CUDA_ASSERT(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
                break;
            case TransferDirection::D2D:
                CUDA_ASSERT(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream));
                break;
        }
        cudaContext->synchronize();
    }
    
    // Measurement
    for (int i = 0; i < iterations; i++) {
        cudaContext->recordStart();
        
        switch (config.direction) {
            case TransferDirection::H2D:
                CUDA_ASSERT(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
                break;
            case TransferDirection::D2H:
                CUDA_ASSERT(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
                break;
            case TransferDirection::D2D:
                CUDA_ASSERT(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream));
                break;
        }
        
        cudaContext->synchronize();
        cudaContext->recordStop();
        
        float elapsedMs = cudaContext->getElapsedTime();
        double latencyUs = elapsedMs * 1000.0;  // Convert to microseconds
        stats.record(latencyUs);
    }
    
    return true;
}
