/*
 * bandwidth_test.h - Bandwidth test runner for cudaperftest
 */

#ifndef BANDWIDTH_TEST_H_
#define BANDWIDTH_TEST_H_

#include "test_runner.h"

// Bandwidth test runner
class BandwidthTestRunner : public TestRunner {
public:
    BandwidthTestRunner(const TestConfig& cfg, const TestEnvironment& env);
    void run() override;
    
protected:
    bool doMemcpy(unsigned long long size, int iterations) override;
    
    // Data integrity verification
    bool verifyTransferData(unsigned long long size, int deviceId);
    bool verifyBidirectionalTransferData(unsigned long long size, int deviceId);
};

#endif  // BANDWIDTH_TEST_H_
