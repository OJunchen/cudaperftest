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
    cudaContext = std::make_unique<CudaContext>(deviceId);
    return true;
}

bool TestRunner::memoryApply(unsigned long long size, int deviceId) {
    bool pinned = (config.hostMemoryType == HostMemoryType::PINNED);
    
    switch (config.direction) {
        case TransferDirection::H2D:
            srcMemory = MemoryFactory::createHostMemory(size, deviceId, pinned);
            dstMemory = MemoryFactory::createDeviceMemory(size, deviceId);
            break;
        case TransferDirection::D2H:
            srcMemory = MemoryFactory::createDeviceMemory(size, deviceId);
            dstMemory = MemoryFactory::createHostMemory(size, deviceId, pinned);
            break;
        case TransferDirection::D2D:
            srcMemory = MemoryFactory::createDeviceMemory(size, config.srcDeviceId);
            dstMemory = MemoryFactory::createDeviceMemory(size, config.dstDeviceId);
            break;
    }
    
    fillPattern(srcMemory->getBuffer(), size, 0xAB);
    return true;
}

void TestRunner::cleanup() {
    srcMemory.reset();
    dstMemory.reset();
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
    std::cout << std::setw(7) << "Device";
    for (const auto& label : sizeLabels) {
        std::cout << std::setw(16) << label;
    }
    std::cout << std::endl;
    
    for (size_t i = 0; i < meanMatrix.size(); i++) {
        std::cout << std::setw(7) << i << " (mean)";
        for (size_t j = 0; j < meanMatrix[i].size() && j < sizeLabels.size(); j++) {
            std::cout << std::setw(16) << std::fixed << std::setprecision(2) << meanMatrix[i][j];
        }
        std::cout << std::endl;
        
        std::cout << std::setw(7) << i << " (P99) ";
        for (size_t j = 0; j < p99Matrix[i].size() && j < sizeLabels.size(); j++) {
            std::cout << std::setw(16) << std::fixed << std::setprecision(2) << p99Matrix[i][j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
