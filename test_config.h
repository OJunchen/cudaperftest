/*
 * test_config.h - Test configuration structures and enums for cudaperftest
 */

#ifndef TEST_CONFIG_H_
#define TEST_CONFIG_H_

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

// Memory type enumeration for Host memory
enum class HostMemoryType {
    PAGEABLE,   // Regular pageable memory (malloc)
    PINNED      // Locked page memory (cudaHostAlloc)
};

// Test mode enumeration
enum class TestMode {
    BANDWIDTH,  // Large data transfer test
    LATENCY     // Small data transfer test
};

// Transfer direction enumeration
enum class TransferDirection {
    H2D,        // Host to Device
    D2H,        // Device to Host
    D2D         // Device to Device
};

// Test configuration structure
struct TestConfig {
    TransferDirection direction;
    TestMode mode;
    HostMemoryType hostMemoryType;
    bool bidirectional;
    int srcDeviceId;  // Source device for D2D
    int dstDeviceId;  // Destination device for D2D
                      // For intra-device copy, set srcDeviceId == dstDeviceId
    
    TestConfig()
        : direction(TransferDirection::H2D),
          mode(TestMode::BANDWIDTH),
          hostMemoryType(HostMemoryType::PAGEABLE),
          bidirectional(false),
          srcDeviceId(0),
          dstDeviceId(1) {}
};

// Test environment structure for environment variable parsing
struct TestEnvironment {
    unsigned int warmupIterations;
    unsigned int bandwidthIterations;
    unsigned int latencyBatchSize;
    std::vector<int> targetDevices;
    bool useAllDevices;
    bool debugMode;
    bool verifyData;  // Data integrity verification flag
    
    TestEnvironment()
        : warmupIterations(3),
          bandwidthIterations(10),
          latencyBatchSize(10),
          useAllDevices(true),
          debugMode(false),
          verifyData(false) {
        parseEnvironmentVariables();
    }
    
    void parseEnvironmentVariables() {
        const char* warmupEnv = std::getenv("PERF_TEST_WARMUP");
        if (warmupEnv != nullptr) {
            try {
                warmupIterations = static_cast<unsigned int>(std::stoul(warmupEnv));
            } catch (...) {}
        }
        
        const char* iterEnv = std::getenv("PERF_TEST_ITERATIONS");
        if (iterEnv != nullptr) {
            try {
                bandwidthIterations = static_cast<unsigned int>(std::stoul(iterEnv));
            } catch (...) {}
        }
        
        const char* batchEnv = std::getenv("PERF_TEST_LATENCY_BATCH");
        if (batchEnv != nullptr) {
            try {
                latencyBatchSize = static_cast<unsigned int>(std::stoul(batchEnv));
                if (latencyBatchSize < 1) latencyBatchSize = 1;
            } catch (...) {}
        }
        
        const char* deviceEnv = std::getenv("PERF_TEST_DEVICE");
        if (deviceEnv != nullptr) {
            targetDevices.clear();
            useAllDevices = false;
            std::string deviceStr(deviceEnv);
            std::stringstream ss(deviceStr);
            std::string token;
            while (std::getline(ss, token, ',')) {
                try {
                    targetDevices.push_back(std::stoi(token));
                } catch (...) {}
            }
            if (targetDevices.empty()) useAllDevices = true;
        }
        
        const char* debugEnv = std::getenv("PERF_TEST_DEBUG");
        if (debugEnv != nullptr) {
            std::string debugStr(debugEnv);
            if (debugStr == "1" || debugStr == "true" || debugStr == "TRUE" ||
                debugStr == "yes" || debugStr == "YES") {
                debugMode = true;
            }
        }
        
        const char* verifyEnv = std::getenv("PERF_TEST_VERIFY");
        if (verifyEnv != nullptr) {
            std::string verifyStr(verifyEnv);
            if (verifyStr == "1" || verifyStr == "true" || verifyStr == "TRUE" ||
                verifyStr == "yes" || verifyStr == "YES") {
                verifyData = true;
            }
        }
    }
};

// Test size configuration
struct TestSizeConfig {
    static constexpr unsigned long long BANDWIDTH_MIN_SIZE = 1ULL * 1024 * 1024;
    static constexpr unsigned long long BANDWIDTH_MAX_SIZE = 1024ULL * 1024 * 1024;
    static constexpr unsigned long long LATENCY_MIN_SIZE = 1;
    static constexpr unsigned long long LATENCY_MAX_SIZE = 64ULL * 1024;
    static constexpr unsigned int LATENCY_BATCH_SIZE = 10;
    static constexpr unsigned int LATENCY_ITERATIONS_PER_BATCH = 100;
};

#endif  // TEST_CONFIG_H_
