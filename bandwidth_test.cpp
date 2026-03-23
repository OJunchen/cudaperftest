/*
 * bandwidth_test.cpp - Bandwidth test implementation for cudaperftest
 */

#include "bandwidth_test.h"
#include <iostream>
#include <iomanip>
#include <vector>

BandwidthTestRunner::BandwidthTestRunner(const TestConfig& cfg, const TestEnvironment& env)
    : TestRunner(cfg, env) {
    testName = "Bandwidth Test - Detailed Statistics";
}

void BandwidthTestRunner::run() {
    std::vector<int> devices = env.useAllDevices ? std::vector<int>{0} : env.targetDevices;
    
    std::vector<std::string> sizeLabels;
    std::vector<std::vector<DetailedStatistics>> statsMatrix;
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
        if (!testInitialize(deviceId)) {
            continue;
        }
        
        std::vector<DetailedStatistics> deviceStats;
        
        for (auto size : sizes) {
            
            bool memoryOk;
            if (config.bidirectional) {
                memoryOk = bidirectionalMemoryApply(size, deviceId);
            } else {
                memoryOk = memoryApply(size, deviceId);
            }
            
            if (!memoryOk) {
                continue;
            }
            
            try {
                stats.reset();
                if (config.bidirectional) {
                    doBidirectionalMemcpy(size, env.bandwidthIterations);
                } else {
                    doMemcpy(size, env.bandwidthIterations);
                }
                
                // Data integrity verification (only if enabled)
                if (env.verifyData) {
                    bool dataOk;
                    if (config.bidirectional) {
                        dataOk = verifyBidirectionalTransferData(size, deviceId);
                    } else {
                        dataOk = verifyTransferData(size, deviceId);
                    }
                    if (!dataOk) {
                        std::cerr << "  Data integrity verification FAILED for size " << size << std::endl;
                    } else {
                        std::cout << "  Data integrity verification PASSED" << std::endl;
                    }
                }
                
                // Calculate detailed statistics
                stats.process();
                DetailedStatistics detailedStat;
                detailedStat.mean = stats.mean();
                detailedStat.stddev = stats.stddev();
                detailedStat.median = stats.median();
                detailedStat.p99 = stats.p99();
                detailedStat.samples = stats.count();
                
                deviceStats.push_back(detailedStat);
            } catch (const std::exception& e) {
                std::cerr << "[DEBUG] Exception caught: " << e.what() << std::endl;
            }
            
            // Only free memory, don't destroy CUDA context
            srcMemory.reset();
            dstMemory.reset();
            bidirSrcMemory.reset();
            bidirDstMemory.reset();
        }
        
        cleanup();
        statsMatrix.push_back(deviceStats);
    }
    
    TestUtils::printDetailedBandwidthStats(statsMatrix, sizeLabels, testName, config);
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
        CUDA_ASSERT(cudaStreamSynchronize(stream));
    }
    
    // Measurement - use CUDA events for accurate timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_ASSERT(cudaEventCreate(&startEvent));
    CUDA_ASSERT(cudaEventCreate(&stopEvent));
    
    for (int i = 0; i < iterations; i++) {
        CUDA_ASSERT(cudaEventRecord(startEvent, stream));
        
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
        
        CUDA_ASSERT(cudaEventRecord(stopEvent, stream));
        CUDA_ASSERT(cudaStreamSynchronize(stream));
        
        float elapsedMs;
        CUDA_ASSERT(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
        double bandwidth = calculateBandwidth(size, elapsedMs);
        stats.record(bandwidth);
    }
    
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    
    return true;
}

bool BandwidthTestRunner::verifyTransferData(unsigned long long size, int deviceId) {
    // For H2D: verify by copying back to host and checking
    // For D2H: dst is host memory, can verify directly
    // For D2D: need to copy to host first
    
    switch (config.direction) {
        case TransferDirection::H2D: {
            // Copy device data back to host for verification
            std::vector<unsigned char> verifyBuffer(size);
            CUDA_ASSERT(cudaMemcpy(verifyBuffer.data(), dstMemory->getBuffer(), size, cudaMemcpyDeviceToHost));
            // Check if pattern matches (0xAB)
            for (size_t i = 0; i < size; i++) {
                if (verifyBuffer[i] != 0xAB) {
                    return false;
                }
            }
            return true;
        }
        case TransferDirection::D2H: {
            // dst is host memory, check directly
            unsigned char* dstPtr = static_cast<unsigned char*>(dstMemory->getBuffer());
            for (size_t i = 0; i < size; i++) {
                if (dstPtr[i] != 0xAB) {
                    return false;
                }
            }
            return true;
        }
        case TransferDirection::D2D: {
            // Copy device data to host for verification
            std::vector<unsigned char> verifyBuffer(size);
            CUDA_ASSERT(cudaSetDevice(config.dstDeviceId));
            CUDA_ASSERT(cudaMemcpy(verifyBuffer.data(), dstMemory->getBuffer(), size, cudaMemcpyDeviceToHost));
            CUDA_ASSERT(cudaSetDevice(deviceId));
            // Check if pattern matches (0xAB)
            for (size_t i = 0; i < size; i++) {
                if (verifyBuffer[i] != 0xAB) {
                    return false;
                }
            }
            return true;
        }
    }
    return false;
}

bool BandwidthTestRunner::verifyBidirectionalTransferData(unsigned long long size, int deviceId) {
    // For bidirectional, verify both directions
    bool forwardOk = true;
    bool reverseOk = true;
    
    switch (config.direction) {
        case TransferDirection::H2D: {
            // Forward: H2D (src->dst), Reverse: D2H (bidirSrc->bidirDst)
            // Verify forward direction
            std::vector<unsigned char> verifyBuffer(size);
            CUDA_ASSERT(cudaMemcpy(verifyBuffer.data(), dstMemory->getBuffer(), size, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < size; i++) {
                if (verifyBuffer[i] != 0xAB) {
                    forwardOk = false;
                    break;
                }
            }
            // Verify reverse direction (bidirDst is host memory)
            unsigned char* bidirDstPtr = static_cast<unsigned char*>(bidirDstMemory->getBuffer());
            for (size_t i = 0; i < size; i++) {
                if (bidirDstPtr[i] != 0xCD) {
                    reverseOk = false;
                    break;
                }
            }
            break;
        }
        case TransferDirection::D2H: {
            // Forward: D2H (src->dst), Reverse: H2D (bidirSrc->bidirDst)
            // Verify forward direction (dst is host memory)
            unsigned char* dstPtr = static_cast<unsigned char*>(dstMemory->getBuffer());
            for (size_t i = 0; i < size; i++) {
                if (dstPtr[i] != 0xAB) {
                    forwardOk = false;
                    break;
                }
            }
            // Verify reverse direction
            std::vector<unsigned char> verifyBuffer(size);
            CUDA_ASSERT(cudaMemcpy(verifyBuffer.data(), bidirDstMemory->getBuffer(), size, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < size; i++) {
                if (verifyBuffer[i] != 0xCD) {
                    reverseOk = false;
                    break;
                }
            }
            break;
        }
        case TransferDirection::D2D: {
            // Forward: D2D (src->dst), Reverse: D2D (bidirSrc->bidirDst)
            // Verify forward direction
            std::vector<unsigned char> verifyBuffer(size);
            CUDA_ASSERT(cudaSetDevice(config.dstDeviceId));
            CUDA_ASSERT(cudaMemcpy(verifyBuffer.data(), dstMemory->getBuffer(), size, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < size; i++) {
                if (verifyBuffer[i] != 0xAB) {
                    forwardOk = false;
                    break;
                }
            }
            // Verify reverse direction
            CUDA_ASSERT(cudaSetDevice(config.srcDeviceId));
            CUDA_ASSERT(cudaMemcpy(verifyBuffer.data(), bidirDstMemory->getBuffer(), size, cudaMemcpyDeviceToHost));
            CUDA_ASSERT(cudaSetDevice(deviceId));
            for (size_t i = 0; i < size; i++) {
                if (verifyBuffer[i] != 0xEF) {
                    reverseOk = false;
                    break;
                }
            }
            break;
        }
    }
    
    return forwardOk && reverseOk;
}
