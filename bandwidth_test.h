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
};

#endif  // BANDWIDTH_TEST_H_
