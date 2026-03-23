/*
 * latency_test.cpp - Latency test implementation for cudaperftest
 */

#include "latency_test.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cmath>

LatencyTestRunner::LatencyTestRunner(const TestConfig& cfg, const TestEnvironment& env)
    : TestRunner(cfg, env) {
    testName = "Latency Test - Detailed Statistics";
}

void LatencyTestRunner::run() {
    std::vector<int> devices = env.useAllDevices ? std::vector<int>{0} : env.targetDevices;
    
    std::vector<std::string> sizeLabels;
    std::vector<std::vector<DetailedStatistics>> statsMatrix;
    
    // Limit latency test to 1B - 64KB range
    std::vector<unsigned long long> sizes = {
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
        1024, 2048, 4096, 8192, 16384, 32768, 65536  // 1B to 64KB
    };
    
    for (auto size : sizes) {
        sizeLabels.push_back(TestUtils::formatSize(size));
    }
    
    for (int deviceId : devices) {
        if (!testInitialize(deviceId)) continue;
        
        std::vector<DetailedStatistics> deviceStats;
        
        for (auto size : sizes) {
            bool memoryOk;
            if (config.bidirectional) {
                memoryOk = bidirectionalMemoryApply(size, deviceId);
            } else {
                memoryOk = memoryApply(size, deviceId);
            }
            
            if (!memoryOk) continue;
            
            try {
                // Run 10 batches, each with 100 iterations
                std::vector<double> batchMeans;
                for (unsigned int batch = 0; batch < TestSizeConfig::LATENCY_BATCH_SIZE; batch++) {
                    stats.reset();
                    if (config.bidirectional) {
                        doBidirectionalMemcpy(size, TestSizeConfig::LATENCY_ITERATIONS_PER_BATCH);
                    } else {
                        doMemcpy(size, TestSizeConfig::LATENCY_ITERATIONS_PER_BATCH);
                    }
                    
                    stats.process();
                    double batchMean = stats.mean();
                    batchMeans.push_back(batchMean);
                }
                
                // Calculate statistics across batches
                DetailedStatistics detailedStat;
                detailedStat.samples = TestSizeConfig::LATENCY_BATCH_SIZE;
                
                // Calculate mean of batch means
                double sum = 0.0;
                for (double val : batchMeans) {
                    sum += val;
                }
                detailedStat.mean = sum / batchMeans.size();
                
                // Calculate stddev
                double varianceSum = 0.0;
                for (double val : batchMeans) {
                    varianceSum += (val - detailedStat.mean) * (val - detailedStat.mean);
                }
                detailedStat.stddev = std::sqrt(varianceSum / batchMeans.size());
                
                // Calculate median
                std::sort(batchMeans.begin(), batchMeans.end());
                size_t n = batchMeans.size();
                detailedStat.median = (n % 2 == 0) ? 
                    (batchMeans[n/2 - 1] + batchMeans[n/2]) / 2.0 : batchMeans[n/2];
                
                // Calculate P99
                size_t p99Idx = static_cast<size_t>(batchMeans.size() * 0.99);
                if (p99Idx >= batchMeans.size()) p99Idx = batchMeans.size() - 1;
                detailedStat.p99 = batchMeans[p99Idx];
                
                deviceStats.push_back(detailedStat);
            } catch (const std::exception& e) {
                std::cerr << "Test failed for size " << size << ": " << e.what() << std::endl;
            }
            
            srcMemory.reset();
            dstMemory.reset();
            bidirSrcMemory.reset();
            bidirDstMemory.reset();
        }
        
        cleanup();
        statsMatrix.push_back(deviceStats);
    }
    
    TestUtils::printDetailedLatencyStats(statsMatrix, sizeLabels, testName, config);
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
