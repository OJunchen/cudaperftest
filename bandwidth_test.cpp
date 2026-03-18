/*
 * bandwidth_test.cpp - Bandwidth test implementation for cudaperftest
 */

#include "bandwidth_test.h"
#include <iostream>
#include <iomanip>

BandwidthTestRunner::BandwidthTestRunner(const TestConfig& cfg, const TestEnvironment& env)
    : TestRunner(cfg, env) {
    testName = "Bandwidth Test";
}

void BandwidthTestRunner::run() {
    std::vector<int> devices = env.useAllDevices ? std::vector<int>{0} : env.targetDevices;
    
    std::vector<std::string> sizeLabels;
    std::vector<std::vector<double>> bandwidthMatrix;
    
    std::vector<unsigned long long> sizes = {
        1ULL * 1024 * 1024,      // 1MB
        4ULL * 1024 * 1024,      // 4MB
        16ULL * 1024 * 1024,     // 16MB
        64ULL * 1024 * 1024,     // 64MB
        256ULL * 1024 * 1024,    // 256MB
        512ULL * 1024 * 1024,    // 512MB
        1024ULL * 1024 * 1024    // 1GB
    };
    
    for (auto size : sizes) {
        sizeLabels.push_back(TestUtils::formatSize(size));
    }
    
    for (int deviceId : devices) {
        if (!testInitialize(deviceId)) continue;
        
        std::vector<double> deviceBandwidth;
        
        for (auto size : sizes) {
            if (!memoryApply(size, deviceId)) continue;
            
            stats.reset();
            doMemcpy(size, env.bandwidthIterations);
            
            double mean, p99;
            calculateLatency(stats, mean, p99);
            deviceBandwidth.push_back(mean);
            
            cleanup();
        }
        
        bandwidthMatrix.push_back(deviceBandwidth);
    }
    
    TestUtils::printBandwidthMatrix(bandwidthMatrix, sizeLabels, testName);
}

bool BandwidthTestRunner::doMemcpy(unsigned long long size, int iterations) {
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
        double bandwidth = calculateBandwidth(size, elapsedMs);
        stats.record(bandwidth);
    }
    
    return true;
}
