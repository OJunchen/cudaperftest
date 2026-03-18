/*
 * test_runner.h - Test runner base class for cudaperftest
 */

#ifndef TEST_RUNNER_H_
#define TEST_RUNNER_H_

#include "test_config.h"
#include "cuda_memory_manager.h"
#include <vector>
#include <memory>
#include <string>

// Performance statistics
class PerformanceStatistics {
private:
    std::vector<double> values;
    std::vector<double> rawValues;

public:
    void record(double value);
    void process();
    void reset();
    double mean() const;
    double median() const;
    double p99() const;
    double stddev() const;
    size_t count() const { return values.size(); }
};

// Test result structures
struct BandwidthTestResult {
    unsigned long long size;
    double bandwidthMean;
    double bandwidthStddev;
    double bandwidthMedian;
    double bandwidthP99;
    std::vector<double> rawSamples;
};

struct LatencyTestResult {
    unsigned long long size;
    double latencyMean;
    double latencyStddev;
    double latencyMedian;
    double latencyP99;
    std::vector<double> rawSamples;
};

// Base test runner class
class TestRunner {
protected:
    TestConfig config;
    TestEnvironment env;
    std::string testName;
    std::unique_ptr<CudaContext> cudaContext;
    std::unique_ptr<BaseMemory> srcMemory;
    std::unique_ptr<BaseMemory> dstMemory;
    PerformanceStatistics stats;

public:
    TestRunner(const TestConfig& cfg, const TestEnvironment& environment);
    virtual ~TestRunner() = default;
    
    // Main entry point
    virtual void run();
    
    // Test phases
    virtual bool testInitialize(int deviceId);
    virtual bool memoryApply(unsigned long long size, int deviceId);
    virtual bool doMemcpy(unsigned long long size, int iterations) = 0;
    virtual void cleanup();
    
    // Calculation methods
    virtual double calculateBandwidth(unsigned long long totalBytes, float elapsedMs);
    virtual void calculateLatency(PerformanceStatistics& stats, double& mean, double& p99);
    
    // Data integrity
    virtual bool verifyDataIntegrity();
    virtual void fillPattern(void* buffer, size_t size, unsigned char pattern);
    virtual bool compareData(void* src, void* dst, size_t size);
    
    // Getters
    const std::string& getTestName() const { return testName; }
    const PerformanceStatistics& getStats() const { return stats; }
};

// Bandwidth test runner
class BandwidthTestRunner : public TestRunner {
public:
    BandwidthTestRunner(const TestConfig& cfg, const TestEnvironment& env);
    
protected:
    bool doMemcpy(unsigned long long size, int iterations) override;
    void runBandwidthTest(int deviceId);
};

// Latency test runner
class LatencyTestRunner : public TestRunner {
public:
    LatencyTestRunner(const TestConfig& cfg, const TestEnvironment& env);
    
protected:
    bool doMemcpy(unsigned long long size, int iterations) override;
    void runLatencyTest(int deviceId);
};

// Utility functions
namespace TestUtils {
    std::string formatSize(unsigned long long size);
    void printBandwidthMatrix(const std::vector<std::vector<double>>& matrix,
                               const std::vector<std::string>& sizeLabels,
                               const std::string& title);
    void printLatencyMatrix(const std::vector<std::vector<double>>& meanMatrix,
                             const std::vector<std::vector<double>>& p99Matrix,
                             const std::vector<std::string>& sizeLabels,
                             const std::string& title);
}

#endif  // TEST_RUNNER_H_
