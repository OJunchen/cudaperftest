/*
 * latency_test.h - Latency test runner for cudaperftest
 */

#ifndef LATENCY_TEST_H_
#define LATENCY_TEST_H_

#include "test_runner.h"

// Latency test runner
class LatencyTestRunner : public TestRunner {
public:
    LatencyTestRunner(const TestConfig& cfg, const TestEnvironment& env);
    void run() override;
    
protected:
    bool doMemcpy(unsigned long long size, int iterations) override;
};

#endif  // LATENCY_TEST_H_
