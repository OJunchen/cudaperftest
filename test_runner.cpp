/*
 * test_runner.cpp - Test runner implementation for cudaperftest
 */

#include "test_runner.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>

// PerformanceStatistics implementation
void PerformanceStatistics::record(double value) {
    rawValues.push_back(value);
}

void PerformanceStatistics::process() {
    if (rawValues.empty()) return;
    
    std::sort(rawValues.begin(), rawValues.end());
    
    size_t n = rawValues.size();
    double med = (n % 2 == 0) ? (rawValues[n/2 - 1] + rawValues[n/2]) / 2.0 : rawValues[n/2];
    
    values.clear();
    for (double val : rawValues) {
        if (val <= med * 2.0) {
            values.push_back(val);
        }
    }
    
    if (values.empty()) {
        values.push_back(med);
    }
    
    std::sort(values.begin(), values.end());
}

void PerformanceStatistics::reset() {
    values.clear();
    rawValues.clear();
}

double PerformanceStatistics::mean() const {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (double v : values) sum += v;
    return sum / values.size();
}

double PerformanceStatistics::median() const {
    if (values.empty()) return 0.0;
    size_t n = values.size();
    return (n % 2 == 0) ? (values[n/2 - 1] + values[n/2]) / 2.0 : values[n/2];
}

double PerformanceStatistics::p99() const {
    if (values.empty()) return 0.0;
    size_t idx = static_cast<size_t>(values.size() * 0.99);
    if (idx >= values.size()) idx = values.size() - 1;
    return values[idx];
}

double PerformanceStatistics::stddev() const {
    if (values.size() < 2) return 0.0;
    double m = mean();
    double sum = 0.0;
    for (double v : values) sum += (v - m) * (v - m);
    return std::sqrt(sum / values.size());
}

// TestRunner implementation
TestRunner::TestRunner(const TestConfig& cfg, const TestEnvironment& environment)
    : config(cfg), env(environment) {}

void TestRunner::run() {
    // To be implemented by derived classes
}

bool TestRunner::testInitialize(int deviceId) {
    // Check if device exists
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceId >= deviceCount) {
        std::cerr << "Error: Device " << deviceId << " not available. Found " 
                  << deviceCount << " device(s)." << std::endl;
        return false;
    }
    
    cudaContext = std::make_unique<CudaContext>(deviceId, config.bidirectional);
    return true;
}

bool TestRunner::memoryApply(unsigned long long size, int deviceId) {
    bool pinned = (config.hostMemoryType == HostMemoryType::PINNED);
    
    // For D2D, validate both devices exist
    if (config.direction == TransferDirection::D2D) {
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        if (config.srcDeviceId >= deviceCount || config.dstDeviceId >= deviceCount) {
            std::cerr << "Error: D2D requires devices " << config.srcDeviceId 
                      << " and " << config.dstDeviceId << ", but only " 
                      << deviceCount << " device(s) available." << std::endl;
            return false;
        }
    }
    
    try {
        switch (config.direction) {
            case TransferDirection::H2D:
                srcMemory = MemoryFactory::createHostMemory(size, deviceId, pinned);
                dstMemory = MemoryFactory::createDeviceMemory(size, deviceId);
                fillPattern(srcMemory->getBuffer(), size, 0xAB);
                break;
            case TransferDirection::D2H:
                srcMemory = MemoryFactory::createDeviceMemory(size, deviceId);
                dstMemory = MemoryFactory::createHostMemory(size, deviceId, pinned);
                fillDevicePattern(srcMemory->getBuffer(), size, 0xAB);
                break;
            case TransferDirection::D2D:
                srcMemory = MemoryFactory::createDeviceMemory(size, config.srcDeviceId);
                dstMemory = MemoryFactory::createDeviceMemory(size, config.dstDeviceId);
                // For D2D, fill pattern on source device with proper context
                CUDA_ASSERT(cudaSetDevice(config.srcDeviceId));
                fillDevicePattern(srcMemory->getBuffer(), size, 0xAB);
                // Restore context to the primary device
                CUDA_ASSERT(cudaSetDevice(deviceId));
                break;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        srcMemory.reset();
        dstMemory.reset();
        return false;
    }
}

bool TestRunner::bidirectionalMemoryApply(unsigned long long size, int deviceId) {
    bool pinned = (config.hostMemoryType == HostMemoryType::PINNED);
    
    try {
        // For bidirectional testing, we need memory for both directions
        // Forward: original direction (H2D, D2H, or D2D)
        // Reverse: opposite direction
        switch (config.direction) {
            case TransferDirection::H2D:
                // Forward: H2D, Reverse: D2H
                srcMemory = MemoryFactory::createHostMemory(size, deviceId, pinned);
                dstMemory = MemoryFactory::createDeviceMemory(size, deviceId);
                bidirSrcMemory = MemoryFactory::createDeviceMemory(size, deviceId);
                bidirDstMemory = MemoryFactory::createHostMemory(size, deviceId, pinned);
                fillPattern(srcMemory->getBuffer(), size, 0xAB);
                fillDevicePattern(bidirSrcMemory->getBuffer(), size, 0xCD);
                break;
            case TransferDirection::D2H:
                // Forward: D2H, Reverse: H2D
                srcMemory = MemoryFactory::createDeviceMemory(size, deviceId);
                dstMemory = MemoryFactory::createHostMemory(size, deviceId, pinned);
                bidirSrcMemory = MemoryFactory::createHostMemory(size, deviceId, pinned);
                bidirDstMemory = MemoryFactory::createDeviceMemory(size, deviceId);
                fillDevicePattern(srcMemory->getBuffer(), size, 0xAB);
                fillPattern(bidirSrcMemory->getBuffer(), size, 0xCD);
                break;
            case TransferDirection::D2D:
                // Forward: D2D (src->dst), Reverse: D2D (dst->src)
                srcMemory = MemoryFactory::createDeviceMemory(size, config.srcDeviceId);
                dstMemory = MemoryFactory::createDeviceMemory(size, config.dstDeviceId);
                bidirSrcMemory = MemoryFactory::createDeviceMemory(size, config.dstDeviceId);
                bidirDstMemory = MemoryFactory::createDeviceMemory(size, config.srcDeviceId);
                // Fill patterns on respective devices
                CUDA_ASSERT(cudaSetDevice(config.srcDeviceId));
                fillDevicePattern(srcMemory->getBuffer(), size, 0xAB);
                fillDevicePattern(bidirDstMemory->getBuffer(), size, 0xEF);
                CUDA_ASSERT(cudaSetDevice(config.dstDeviceId));
                fillDevicePattern(dstMemory->getBuffer(), size, 0xCD);
                fillDevicePattern(bidirSrcMemory->getBuffer(), size, 0xEF);
                CUDA_ASSERT(cudaSetDevice(deviceId));
                break;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Bidirectional memory allocation failed: " << e.what() << std::endl;
        srcMemory.reset();
        dstMemory.reset();
        bidirSrcMemory.reset();
        bidirDstMemory.reset();
        return false;
    }
}

bool TestRunner::doBidirectionalMemcpy(unsigned long long size, int iterations) {
    void* src = srcMemory->getBuffer();
    void* dst = dstMemory->getBuffer();
    void* bidirSrc = bidirSrcMemory->getBuffer();
    void* bidirDst = bidirDstMemory->getBuffer();
    cudaStream_t streamForward = cudaContext->getStreamForward();
    cudaStream_t streamReverse = cudaContext->getStreamReverse();
    
    // Warmup
    for (unsigned int w = 0; w < env.warmupIterations; w++) {
        // Forward transfer
        switch (config.direction) {
            case TransferDirection::H2D:
                CUDA_ASSERT(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, streamForward));
                break;
            case TransferDirection::D2H:
                CUDA_ASSERT(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, streamForward));
                break;
            case TransferDirection::D2D:
                CUDA_ASSERT(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, streamForward));
                break;
        }
        // Reverse transfer
        switch (config.direction) {
            case TransferDirection::H2D:
                // Reverse: D2H
                CUDA_ASSERT(cudaMemcpyAsync(bidirDst, bidirSrc, size, cudaMemcpyDeviceToHost, streamReverse));
                break;
            case TransferDirection::D2H:
                // Reverse: H2D
                CUDA_ASSERT(cudaMemcpyAsync(bidirDst, bidirSrc, size, cudaMemcpyHostToDevice, streamReverse));
                break;
            case TransferDirection::D2D:
                // Reverse: D2D (opposite direction)
                CUDA_ASSERT(cudaMemcpyAsync(bidirDst, bidirSrc, size, cudaMemcpyDeviceToDevice, streamReverse));
                break;
        }
        cudaContext->synchronizeAllStreams();
    }
    
    // Measurement
    for (int i = 0; i < iterations; i++) {
        cudaContext->recordBidirectionalStart();
        
        // Forward transfer
        switch (config.direction) {
            case TransferDirection::H2D:
                CUDA_ASSERT(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, streamForward));
                break;
            case TransferDirection::D2H:
                CUDA_ASSERT(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, streamForward));
                break;
            case TransferDirection::D2D:
                CUDA_ASSERT(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, streamForward));
                break;
        }
        // Reverse transfer
        switch (config.direction) {
            case TransferDirection::H2D:
                CUDA_ASSERT(cudaMemcpyAsync(bidirDst, bidirSrc, size, cudaMemcpyDeviceToHost, streamReverse));
                break;
            case TransferDirection::D2H:
                CUDA_ASSERT(cudaMemcpyAsync(bidirDst, bidirSrc, size, cudaMemcpyHostToDevice, streamReverse));
                break;
            case TransferDirection::D2D:
                CUDA_ASSERT(cudaMemcpyAsync(bidirDst, bidirSrc, size, cudaMemcpyDeviceToDevice, streamReverse));
                break;
        }
        
        cudaContext->synchronizeAllStreams();
        cudaContext->recordBidirectionalStop();
        
        float elapsedMs = cudaContext->getBidirectionalElapsedTime();
        // Bidirectional bandwidth counts total bytes in both directions
        double bandwidth = calculateBandwidth(size * 2, elapsedMs);
        stats.record(bandwidth);
    }
    
    return true;
}

void TestRunner::cleanup() {
    srcMemory.reset();
    dstMemory.reset();
    bidirSrcMemory.reset();
    bidirDstMemory.reset();
    cudaContext.reset();
}

double TestRunner::calculateBandwidth(unsigned long long totalBytes, float elapsedMs) {
    double seconds = elapsedMs / 1000.0;
    double bytes = static_cast<double>(totalBytes);
    return (bytes / seconds) / (1024.0 * 1024.0 * 1024.0);  // GB/s
}

void TestRunner::calculateLatency(PerformanceStatistics& stats, double& mean, double& p99) {
    stats.process();
    mean = stats.mean();
    p99 = stats.p99();
}

bool TestRunner::verifyDataIntegrity() {
    return compareData(srcMemory->getBuffer(), dstMemory->getBuffer(), srcMemory->getBufferSize());
}

void TestRunner::fillPattern(void* buffer, size_t size, unsigned char pattern) {
    unsigned char* ptr = static_cast<unsigned char*>(buffer);
    for (size_t i = 0; i < size; i++) {
        ptr[i] = pattern;
    }
}

void TestRunner::fillDevicePattern(void* buffer, size_t size, unsigned char pattern) {
    CUDA_ASSERT(cudaMemset(buffer, pattern, size));
}

bool TestRunner::compareData(void* src, void* dst, size_t size) {
    unsigned char* srcPtr = static_cast<unsigned char*>(src);
    unsigned char* dstPtr = static_cast<unsigned char*>(dst);
    for (size_t i = 0; i < size; i++) {
        if (srcPtr[i] != dstPtr[i]) return false;
    }
    return true;
}

// TestUtils implementation
std::string TestUtils::formatSize(unsigned long long size) {
    std::ostringstream oss;
    if (size >= 1024ULL * 1024 * 1024) {
        oss << std::fixed << std::setprecision(1) << (size / (1024.0 * 1024 * 1024)) << " GB";
    } else if (size >= 1024 * 1024) {
        oss << std::fixed << std::setprecision(1) << (size / (1024.0 * 1024)) << " MB";
    } else if (size >= 1024) {
        oss << std::fixed << std::setprecision(1) << (size / 1024.0) << " KB";
    } else {
        oss << size << " B";
    }
    return oss.str();
}

void TestUtils::printBandwidthMatrix(const std::vector<std::vector<double>>& matrix,
                                      const std::vector<std::string>& sizeLabels,
                                      const std::string& title) {
    std::cout << title << " (GB/s)" << std::endl;
    std::cout << std::setw(7) << "Device";
    for (const auto& label : sizeLabels) {
        std::cout << std::setw(10) << label;
    }
    std::cout << std::endl;
    
    for (size_t i = 0; i < matrix.size(); i++) {
        std::cout << std::setw(7) << i;
        for (size_t j = 0; j < matrix[i].size() && j < sizeLabels.size(); j++) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << matrix[i][j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void TestUtils::printLatencyMatrix(const std::vector<std::vector<double>>& meanMatrix,
                                    const std::vector<std::vector<double>>& p99Matrix,
                                    const std::vector<std::string>& sizeLabels,
                                    const std::string& title) {
    std::cout << title << " (us)" << std::endl;
    std::cout << std::setw(9) << "Device";
    for (const auto& label : sizeLabels) {
        std::cout << std::setw(12) << label;
    }
    std::cout << std::endl;
    
    for (size_t i = 0; i < meanMatrix.size(); i++) {
        std::cout << std::setw(9) << (std::to_string(i) + " (mean)");
        for (size_t j = 0; j < meanMatrix[i].size() && j < sizeLabels.size(); j++) {
            std::cout << std::setw(12) << std::fixed << std::setprecision(2) << meanMatrix[i][j];
        }
        std::cout << std::endl;
        
        std::cout << std::setw(9) << (std::to_string(i) + " (P99)");
        for (size_t j = 0; j < p99Matrix[i].size() && j < sizeLabels.size(); j++) {
            std::cout << std::setw(12) << std::fixed << std::setprecision(2) << p99Matrix[i][j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

std::string TestUtils::getTransferTypeString(const TestConfig& config) {
    std::string direction;
    switch (config.direction) {
        case TransferDirection::H2D:
            direction = "H2D";
            break;
        case TransferDirection::D2H:
            direction = "D2H";
            break;
        case TransferDirection::D2D:
            direction = "D2D";
            break;
    }
    
    std::string memoryType;
    if (config.direction != TransferDirection::D2D) {
        memoryType = (config.hostMemoryType == HostMemoryType::PINNED) ? "Pinned" : "Pageable";
    }
    
    if (config.bidirectional) {
        if (config.direction == TransferDirection::D2D) {
            return "Bidirectional " + direction;
        } else {
            return "Bidirectional " + direction + " (" + memoryType + " -> DDR)";
        }
    } else {
        if (config.direction == TransferDirection::D2D) {
            return "Unidirectional " + direction;
        } else {
            return "Unidirectional " + direction + " (" + memoryType + " -> DDR)";
        }
    }
}

void TestUtils::printDetailedLatencyStats(const std::vector<std::vector<DetailedStatistics>>& statsMatrix,
                                           const std::vector<std::string>& sizeLabels,
                                           const std::string& title,
                                           const TestConfig& config) {
    std::string transferType = getTransferTypeString(config);
    std::cout << transferType << " - " << title << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << std::setw(10) << "Size" 
              << std::setw(15) << "Mean(us)" 
              << std::setw(15) << "Stddev" 
              << std::setw(15) << "Median" 
              << std::setw(15) << "P99" 
              << std::setw(10) << "Samples" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    
    for (size_t i = 0; i < statsMatrix.size(); i++) {
        for (size_t j = 0; j < statsMatrix[i].size() && j < sizeLabels.size(); j++) {
            const auto& stat = statsMatrix[i][j];
            std::cout << std::setw(10) << sizeLabels[j]
                      << std::setw(15) << std::fixed << std::setprecision(2) << stat.mean
                      << std::setw(15) << std::fixed << std::setprecision(2) << stat.stddev
                      << std::setw(15) << std::fixed << std::setprecision(2) << stat.median
                      << std::setw(15) << std::fixed << std::setprecision(2) << stat.p99
                      << std::setw(10) << stat.samples << std::endl;
        }
    }
    std::cout << std::endl;
}

void TestUtils::printDetailedBandwidthStats(const std::vector<std::vector<DetailedStatistics>>& statsMatrix,
                                             const std::vector<std::string>& sizeLabels,
                                             const std::string& title,
                                             const TestConfig& config) {
    std::string transferType = getTransferTypeString(config);
    std::cout << transferType << " - " << title << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << std::setw(10) << "Size" 
              << std::setw(15) << "Mean(GB/s)" 
              << std::setw(15) << "Stddev" 
              << std::setw(15) << "Median" 
              << std::setw(15) << "P99" 
              << std::setw(10) << "Samples" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    
    for (size_t i = 0; i < statsMatrix.size(); i++) {
        for (size_t j = 0; j < statsMatrix[i].size() && j < sizeLabels.size(); j++) {
            const auto& stat = statsMatrix[i][j];
            std::cout << std::setw(10) << sizeLabels[j]
                      << std::setw(15) << std::fixed << std::setprecision(2) << stat.mean
                      << std::setw(15) << std::fixed << std::setprecision(2) << stat.stddev
                      << std::setw(15) << std::fixed << std::setprecision(2) << stat.median
                      << std::setw(15) << std::fixed << std::setprecision(2) << stat.p99
                      << std::setw(10) << stat.samples << std::endl;
        }
    }
    std::cout << std::endl;
}
