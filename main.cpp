/*
 * main.cpp - Entry point for cudaperftest
 * 
 * Command line:
 *   -t <h2d|d2d|d2h>     : Test target (required)
 *   -B                   : Bandwidth test (default)
 *   -L                   : Latency test
 *   -p <0|1>             : Host memory type (0=pageable, 1=pinned), default 0
 *   -d <0|1>             : Direction (0=unidirectional, 1=bidirectional), default 0
 *   --src-dev <id>       : Source device for D2D, default 0
 *   --dst-dev <id>       : Destination device for D2D, default 1
 * 
 * Environment variables:
 *   PERF_TEST_WARMUP     : Warmup iterations (default: 3)
 *   PERF_TEST_ITERATIONS : Bandwidth test iterations (default: 10)
 *   PERF_TEST_DEVICE     : Target devices, e.g., "0,1,2,3" (default: all)
 */

#include <iostream>
#include <memory>
#include <cstring>
#include "test_config.h"
#include "test_runner.h"

class CommandLineParser {
private:
    int argc;
    char** argv;

public:
    CommandLineParser(int argc, char** argv) : argc(argc), argv(argv) {}
    
    const char* getOptionValue(const char* option) {
        for (int i = 1; i < argc - 1; i++) {
            if (strcmp(argv[i], option) == 0) {
                return argv[i + 1];
            }
        }
        return nullptr;
    }
    
    bool hasOption(const char* option) {
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], option) == 0) {
                return true;
            }
        }
        return false;
    }
};

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -t <h2d|d2d|d2h>  Test target (required)" << std::endl;
    std::cout << "                    h2d: Host to Device" << std::endl;
    std::cout << "                    d2h: Device to Host" << std::endl;
    std::cout << "                    d2d: Device to Device" << std::endl;
    std::cout << "  -B                Bandwidth test (default)" << std::endl;
    std::cout << "  -L                Latency test" << std::endl;
    std::cout << "  -p <0|1>          Host memory type (0=pageable, 1=pinned), default: 0" << std::endl;
    std::cout << "  -d <0|1>          Direction (0=unidirectional, 1=bidirectional), default: 0" << std::endl;
    std::cout << "  --src-dev <id>    Source device for D2D, default: 0" << std::endl;
    std::cout << "  --dst-dev <id>    Destination device for D2D, default: 1" << std::endl;
    std::cout << "  -h, --help        Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Environment Variables:" << std::endl;
    std::cout << "  PERF_TEST_WARMUP      Warmup iterations (default: 3)" << std::endl;
    std::cout << "  PERF_TEST_ITERATIONS  Bandwidth test iterations (default: 10)" << std::endl;
    std::cout << "  PERF_TEST_DEVICE      Target devices, e.g., \"0,1,2,3\" (default: all)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << programName << " -t h2d -B -p 1       # H2D bandwidth test with pinned memory" << std::endl;
    std::cout << "  " << programName << " -t d2h -L -p 1       # D2H latency test with pinned memory" << std::endl;
    std::cout << "  " << programName << " -t d2d -B --src-dev 0 --dst-dev 1  # D2D bandwidth test" << std::endl;
}

bool parseArguments(int argc, char** argv, TestConfig& config) {
    CommandLineParser parser(argc, argv);
    
    if (parser.hasOption("-h") || parser.hasOption("--help") || argc == 1) {
        printUsage(argv[0]);
        return false;
    }
    
    const char* targetStr = parser.getOptionValue("-t");
    const char* hostMemStr = parser.getOptionValue("-p");
    const char* bidirStr = parser.getOptionValue("-d");
    const char* srcDevStr = parser.getOptionValue("--src-dev");
    const char* dstDevStr = parser.getOptionValue("--dst-dev");
    
    if (!targetStr) {
        std::cerr << "Error: Must specify -t <target>" << std::endl;
        printUsage(argv[0]);
        return false;
    }
    
    if (strcmp(targetStr, "h2d") == 0) {
        config.direction = TransferDirection::H2D;
    } else if (strcmp(targetStr, "d2h") == 0) {
        config.direction = TransferDirection::D2H;
    } else if (strcmp(targetStr, "d2d") == 0) {
        config.direction = TransferDirection::D2D;
    } else {
        std::cerr << "Error: Invalid target. Use h2d, d2h, or d2d." << std::endl;
        return false;
    }
    
    if (parser.hasOption("-L")) {
        config.mode = TestMode::LATENCY;
    } else {
        config.mode = TestMode::BANDWIDTH;
    }
    
    if (hostMemStr) {
        int hostMemInt = atoi(hostMemStr);
        if (hostMemInt == 0) {
            config.hostMemoryType = HostMemoryType::PAGEABLE;
        } else if (hostMemInt == 1) {
            config.hostMemoryType = HostMemoryType::PINNED;
        } else {
            std::cerr << "Error: Invalid host memory type. Use 0 (pageable) or 1 (pinned)." << std::endl;
            return false;
        }
    }
    
    if (bidirStr) {
        config.bidirectional = (atoi(bidirStr) == 1);
    }
    
    if (srcDevStr) {
        config.srcDeviceId = atoi(srcDevStr);
    }
    
    if (dstDevStr) {
        config.dstDeviceId = atoi(dstDevStr);
    }
    
    return true;
}

int main(int argc, char** argv) {
    TestConfig config;
    TestEnvironment env;
    
    if (!parseArguments(argc, argv, config)) {
        return 1;
    }
    
    try {
        std::unique_ptr<TestRunner> runner;
        
        if (config.mode == TestMode::BANDWIDTH) {
            runner = std::make_unique<BandwidthTestRunner>(config, env);
        } else {
            runner = std::make_unique<LatencyTestRunner>(config, env);
        }
        
        runner->run();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
