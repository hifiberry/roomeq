# RoomEQ Rust Optimizer & FFT Analyzer

A high-performance Rust implementation of the RoomEQ optimization algorithm and FFT analysis tools for real-time audio processing and room acoustic analysis.

## Binaries

### roomeq-optimizer
The main room EQ optimizer that finds optimal DSP filter parameters to correct room acoustics.

### roomeq-analyzer (NEW!)
High-performance FFT analysis tool - a Rust port of HiFiBerry's original C FFT analyzer for audio measurements and frequency response analysis.

## FFT Analyzer Features

- **High Performance**: Optimized Rust implementation with release-mode builds (< 1 second analysis)
- **Multi-file Support**: Analyze multiple WAV files against a reference
- **Logarithmic Frequency Binning**: Reduces FFT data to manageable points with perceptually relevant spacing  
- **Reference Comparison**: Compare recordings against known reference signals
- **CSV Export**: Output frequency response data in standard CSV format
- **Configurable Analysis**: Adjustable frequency range, output points, and verbosity

## FFT Analyzer Quick Start

```bash
# Build the analyzer
cargo build --bin roomeq-analyzer --release

# Basic analysis
./target/release/roomeq-analyzer \
  -r reference.wav \
  -n 64 \
  -o output.csv \
  recording1.wav recording2.wav

# Advanced analysis with custom parameters
./target/release/roomeq-analyzer \
  --reference signal.wav \
  --verbose 2 \
  --points 128 \
  --freq-min 20 \
  --freq-max 20000 \
  --output room_response.csv \
  measurement1.wav measurement2.wav
```

## RoomEQ Optimizer Features

- **High Performance**: Optimized Rust implementation for fast processing
- **Real-time Progress**: Step-by-step optimization progress via JSON output
- **Compatible API**: Same input/output format as Python implementation
- **Biquad Filters**: Full support for HP/LP/BP/Notch/Peaking/Shelf filters
- **Target Curves**: Support for all target curve types with weighting
- **Optimizer Presets**: Compatible with all Python optimizer presets

## Building

### Prerequisites
- Rust 1.70 or later
- Cargo package manager

### Build Instructions

```bash
# Navigate to the Rust directory
cd src/rust

# Build in release mode (optimized)
cargo build --release

# The binary will be created at: target/release/roomeq-optimizer
```

### Development Build (Debug)
```bash
cargo build
# Debug binary at: target/debug/roomeq-optimizer
```

## Usage

The Rust optimizer reads optimization jobs from stdin and outputs progress steps to stdout in JSON format.

### Input Format (JSON via stdin)

```json
{
  "measured_curve": {
    "frequencies": [20.0, 25.0, 31.5, ...],
    "magnitudes_db": [2.1, 1.8, 0.9, ...]
  },
  "target_curve": {
    "name": "Weighted Flat",
    "description": "Flat response with frequency-dependent correction weights",
    "expert": true,
    "curve": [
      {"frequency": 20.0, "target_db": 0.0, "weight": [1.0, 0.3]},
      {"frequency": 100.0, "target_db": 0.0, "weight": [1.0, 0.6]},
      {"frequency": 25000.0, "target_db": 0.0, "weight": null}
    ]
  },
  "optimizer_params": {
    "name": "Default",
    "description": "Balanced optimization with moderate Q values",
    "qmax": 10.0,
    "mindb": -10.0,
    "maxdb": 3.0,
    "add_highpass": true
  },
  "sample_rate": 48000.0,
  "filter_count": 8
}
```

### Output Format (JSON Lines to stdout)

Each optimization step outputs one JSON line:

```json
{
  "step": 1,
  "filters": [
    {
      "filter_type": "eq",
      "frequency": 120.0,
      "q": 3.2,
      "gain_db": -4.5,
      "description": "Peaking EQ 120.0Hz -4.5dB",
      "coefficients": {
        "b": [1.0123, -1.9876, 0.9753],
        "a": [1.0, -1.9876, 0.9876]
      }
    }
  ],
  "corrected_response": {
    "frequencies": [20.0, 25.0, 31.5, ...],
    "magnitudes_db": [1.8, 1.5, 0.6, ...]
  },
  "residual_error": 2.34,
  "message": "Added filter 1 at 120.0Hz (-4.5dB)",
  "progress_percent": 12.5
}
```

### Command Line Usage

```bash
# Run with input from file
./target/release/roomeq-optimizer < optimization_job.json

# Run with piped input
echo '{"measured_curve":...}' | ./target/release/roomeq-optimizer

# Run with Python demo
python3 demo_rust_optimizer.py
```

## Target Curve Weights

The weight parameter in target curves controls optimization behavior:

- **`null`**: Default weighting (1.0, 1.0)
- **Single value**: `0.5` → Equal weighting for boosts and cuts
- **Array of 2**: `[1.0, 0.3]` → Different weights for positive/negative corrections
- **Array of 3**: `[1.0, 0.3, 1000.0]` → With frequency cutoff parameter

## Filter Types

Supported biquad filter types:

- **`hp`**: High-pass filter (frequency, Q)
- **`eq`**: Peaking EQ filter (frequency, Q, gain_dB)  
- **`ls`**: Low shelf filter (frequency, Q, gain_dB)
- **`hs`**: High shelf filter (frequency, Q, gain_dB)

## Example: Basic Usage

```bash
# Create optimization job
cat > job.json << 'EOF'
{
  "measured_curve": {
    "frequencies": [20, 100, 1000, 10000, 20000],
    "magnitudes_db": [2.0, -3.0, 1.5, -2.0, -4.0]
  },
  "target_curve": {
    "name": "Flat Response",
    "description": "Flat frequency response",
    "expert": false,
    "curve": [
      {"frequency": 20.0, "target_db": 0.0, "weight": null},
      {"frequency": 25000.0, "target_db": 0.0, "weight": null}
    ]
  },
  "optimizer_params": {
    "name": "Default",
    "description": "Balanced optimization",
    "qmax": 10.0,
    "mindb": -10.0,
    "maxdb": 3.0,
    "add_highpass": true
  },
  "sample_rate": 48000.0,
  "filter_count": 6
}
EOF

# Run optimization
./target/release/roomeq-optimizer < job.json
```

## Python Integration

Use the provided `demo_rust_optimizer.py` to see how to integrate with Python:

```python
import json
import subprocess

# Create optimization job
job = create_optimization_job(
    curve_type="weighted_flat",
    preset_type="default",
    filter_count=8,
    sample_rate=48000.0
)

# Run Rust optimizer
steps = run_rust_optimizer(job, "./target/release/roomeq-optimizer")

# Process results
for step in steps:
    print(f"Step {step['step']}: Error = {step['residual_error']:.2f} dB")
```

## Performance

The Rust implementation provides significant performance improvements:

- **High performance** room EQ optimization
- **Low memory usage** through efficient data structures  
- **Real-time capable** for live audio processing
- **Parallel optimization** potential for multiple channels

## Testing

Run the included tests:

```bash
cargo test
```

Test with sample data:

```bash
python3 demo_rust_optimizer.py ./target/release/roomeq-optimizer
```

## Integration with Python RoomEQ

The Rust optimizer can be integrated into the existing Python RoomEQ system:

1. **Subprocess approach**: Call Rust binary from Python (demonstrated)
2. **PyO3 bindings**: Create Python extension module (future enhancement)
3. **REST API**: Run as microservice for high-throughput processing

## Error Handling

- **Invalid JSON input**: Returns exit code 1 with error message to stderr
- **Missing parameters**: Uses sensible defaults where possible
- **Optimization failures**: Reports error in final step output
- **File not found**: Clear error message for missing binary

## Logging

Set log level with environment variable:

```bash
RUST_LOG=info ./target/release/roomeq-optimizer < job.json
RUST_LOG=debug ./target/release/roomeq-optimizer < job.json
```

## Optimization Algorithm

The Rust implementation uses a simplified but effective optimization approach:

1. **Frequency Analysis**: Generate log-spaced optimization frequencies
2. **Error Detection**: Identify frequencies with largest target deviations
3. **Filter Selection**: Choose appropriate filter type (EQ/shelf) based on frequency and gain
4. **Progressive Correction**: Add filters one by one, minimizing overall error
5. **Real-time Feedback**: Output JSON step data for each filter addition

## Future Enhancements

- **Advanced Algorithms**: Implement more sophisticated optimization methods
- **Parallel Processing**: Multi-threaded optimization for complex curves
- **SIMD Optimization**: Vectorized frequency response calculations  
- **Memory Optimization**: Further reduce memory footprint
- **PyO3 Bindings**: Direct Python integration without subprocess overhead
