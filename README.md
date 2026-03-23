# cudaperftest

A comprehensive CUDA performance testing tool for measuring bandwidth and latency of various memory transfer patterns between host and device, as well as device-to-device transfers.

## Features

- **Bandwidth Testing**: Measure throughput for large data transfers
- **Latency Testing**: Measure timing for small data transfers
- **Multiple Transfer Types**:
  - Host to Device (H2D)
  - Device to Host (D2H)
  - Device to Device (D2D) - both cross-device and intra-device
- **Flexible Configuration**:
  - Support for both pageable and pinned host memory
  - Unidirectional and bidirectional transfers
  - Configurable warmup and test iterations
  - Data integrity verification
- **Multi-GPU Support**: Test across multiple GPU devices

## Requirements

- CUDA Toolkit 11.0 or higher
- C++14 compatible compiler (GCC 7.x or above recommended)
- CMake 3.10 or higher
- NVIDIA GPU with compute capability 7.0 or higher

## Building

### Prerequisites

Ensure CUDA is installed and the `nvcc` compiler is in your PATH.

### Build Instructions

```bash
cd cudaperftest
mkdir build && cd build
cmake ..
make
```

The executable `cudaperftest` will be created in the build directory.

### Customizing CUDA Architecture

Edit `CMakeLists.txt` to adjust the CUDA architectures based on your GPU:

```cmake
set_target_properties(cudaperftest PROPERTIES
    CUDA_ARCHITECTURES "70;80"  # Add your GPU architectures here
)
```

## Usage

### Command Line Options

```
Usage: ./cudaperftest [options]

Options:
  -t <h2d|d2d|d2h>  Test target (required)
                    h2d: Host to Device
                    d2h: Device to Host
                    d2d: Device to Device
  -B                Bandwidth test (default)
  -L                Latency test
  -p <0|1>          Host memory type (0=pageable, 1=pinned), default: 0
  -d <0|1>          Direction (0=unidirectional, 1=bidirectional), default: 0
  --src-dev <id>    Source device for D2D, default: 0
  --dst-dev <id>    Destination device for D2D, default: 1
                    Use same ID for intra-device copy (e.g., both 0)
  -h, --help        Show this help message
```

### Environment Variables

- `PERF_TEST_WARMUP`: Warmup iterations (default: 3)
- `PERF_TEST_ITERATIONS`: Bandwidth test iterations (default: 10)
- `PERF_TEST_DEVICE`: Target devices, e.g., "0,1,2,3" (default: all)
- `PERF_TEST_VERIFY`: Enable data integrity verification (default: disabled)

## Examples

### Host to Device Bandwidth Test

```bash
# Basic H2D bandwidth test with pageable memory
./cudaperftest -t h2d -B

# H2D bandwidth test with pinned memory (recommended for better performance)
./cudaperftest -t h2d -B -p 1
```

### Device to Host Latency Test

```bash
# D2H latency test with pinned memory
./cudaperftest -t d2h -L -p 1
```

### Device to Device Bandwidth Tests

```bash
# Cross-device D2D bandwidth test (device 0 to device 1)
./cudaperftest -t d2d -B --src-dev 0 --dst-dev 1

# Intra-device D2D bandwidth test (same device)
./cudaperftest -t d2d -B --src-dev 0 --dst-dev 0

# Bidirectional D2D test
./cudaperftest -t d2d -B --src-dev 0 --dst-dev 1 -d 1
```

### Advanced Configuration

```bash
# Custom iterations with data verification
PERF_TEST_ITERATIONS=20 PERF_TEST_WARMUP=5 PERF_TEST_VERIFY=1 \
./cudaperftest -t d2d -B --src-dev 0 --dst-dev 0

# Test on specific devices
PERF_TEST_DEVICE="0,2,4" ./cudaperftest -t h2d -B -p 1
```

## Understanding D2D Tests

### Cross-Device D2D
When `--src-dev` and `--dst-dev` specify different devices, the test measures bandwidth across GPU interconnects (PCIe, NVLink, etc.). This typically yields 30-100 GB/s depending on the interconnect type.

```bash
./cudaperftest -t d2d -B --src-dev 0 --dst-dev 1
```

### Intra-Device D2D
When both device IDs are the same, the test measures bandwidth within a single GPU's memory. This tests the GPU's internal memory bandwidth and can achieve 1-2 TB/s on modern GPUs.

```bash
./cudaperftest -t d2d -B --src-dev 0 --dst-dev 0
```

This is comparable to the `device_local_copy` test in the [nvbandwidth](https://github.com/NVIDIA/nvbandwidth) tool.

## Output Format

### Bandwidth Test Output

```
Bandwidth Test (GB/s)
Device      1 MB      4 MB     16 MB     64 MB    256 MB    512 MB      1 GB
0      25.50    26.10    26.30    26.40    26.45    26.48    26.50
```

### Latency Test Output

```
Latency Test (us)
Device        1 B       4 B      16 B      64 B     256 B      1 KB      4 KB     16 KB
0 (mean)    2.50     2.55     2.60     2.65     2.70     2.80     3.00     3.50
0 (P99)     3.00     3.10     3.20     3.30     3.40     3.60     4.00     4.50
```

## Performance Tips

1. **Use Pinned Memory**: For H2D and D2H transfers, pinned memory (`-p 1`) typically provides better performance than pageable memory.

2. **Warmup Iterations**: Increase warmup iterations for more stable results, especially on systems with dynamic frequency scaling.

3. **Data Verification**: Enable `PERF_TEST_VERIFY=1` to ensure data integrity during testing, though this may slightly impact performance.

4. **Intra-Device vs Cross-Device**: Remember that intra-device D2D tests measure GPU internal memory bandwidth (much higher), while cross-device tests measure interconnect bandwidth.

## Project Structure

```
cudaperftest/
├── main.cpp                    # Entry point and command line parsing
├── test_config.h               # Test configuration structures
├── test_runner.h/cpp           # Base test runner implementation
├── bandwidth_test.h/cpp        # Bandwidth test implementation
├── latency_test.h/cpp          # Latency test implementation
├── cuda_memory_manager.h/cpp   # CUDA memory and context management
└── CMakeLists.txt              # Build configuration
```

## Technical Details

### Measurement Methodology

- **Bandwidth Tests**: Use CUDA events to measure transfer time for large buffers (1MB to 1GB)
- **Latency Tests**: Use high-resolution CPU timers for small transfers (1B to 8MB)
- **Timing**: Multiple iterations are performed, and median/P99 statistics are reported

### Memory Management

- Device memory is allocated using `cudaMalloc`
- Host memory can be allocated as pageable (`malloc`) or pinned (`cudaHostAlloc`)
- CUDA streams are used for asynchronous transfers
- Proper device context management ensures correct behavior in multi-GPU scenarios

## Comparison with nvbandwidth

cudaperftest provides similar functionality to [NVIDIA's nvbandwidth](https://github.com/NVIDIA/nvbandwidth) tool but with some differences:

- **Simpler Interface**: Easier to use for common test scenarios
- **Flexible Configuration**: More control over test parameters via environment variables
- **Intra-Device D2D**: Explicit support for testing GPU internal memory bandwidth
- **Modern C++**: Uses C++14 features for cleaner code

For comprehensive testing of various copy patterns (CE, SM, multicast, etc.), consider using nvbandwidth. For focused bandwidth and latency testing with flexible configuration, cudaperftest provides a convenient alternative.

## License

This project is provided as-is for performance testing purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Inspired by NVIDIA's [nvbandwidth](https://github.com/NVIDIA/nvbandwidth) tool
- Built with CUDA and modern C++
