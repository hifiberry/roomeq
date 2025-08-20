# RoomEQ Optimizer Documentation

## Overview

The RoomEQ optimizer is a high-performance, Rust-based room equalization system that automatically generates parametric EQ filters to correct room acoustics. It uses advanced algorithms including frequency deduplication, adaptive high-pass filtering, and frequency weighting to create optimal correction filters.

## System Architecture

### Backend: Rust Optimizer (v0.6.0)
- **Algorithm**: Brute-force search with intelligent optimization
- **Output**: Real-time streaming without buffering
- **Binary Location**: `src/rust/target/release/roomeq-optimizer`

### Frontend: Python REST API (v0.7.0)
- **Integration**: Streams Rust optimizer output directly to clients
- **No Buffering**: Real-time progress updates via Server-Sent Events
- **Error Handling**: Comprehensive error reporting and validation

## Core Algorithms

### 1. Usable Frequency Range Detection

The optimizer automatically determines which frequencies can be effectively corrected:

**Algorithm**:
1. Calculate average magnitude in reference range (200-8000 Hz)
2. Apply -8 dB threshold relative to this average
3. Find lowest and highest frequencies above threshold
4. Result: `f_low` to `f_high` range for correction

**Purpose**: Prevents optimization at frequencies where room response is too low to correct meaningfully.

**Example**: If room has severe nulls below 40 Hz and above 15 kHz, optimizer focuses on 40-15000 Hz range.

### 2. Intelligent Frequency Candidate Generation

Creates an optimal set of frequencies for filter placement:

**Process**:
1. **Measured Frequencies**: Start with original measurement points
2. **Interpolated Frequencies**: Add geometric mean between adjacent points
   - Formula: `f_interp = sqrt(f1 * f2)`
   - Example: Between 1000 Hz and 2000 Hz, adds 1414 Hz
3. **Boundary Exclusion**: Remove frequencies too close to f_low/f_high (5% margin)
4. **Result**: Typically ~58 candidates from ~31 measured frequencies

**Benefits**: 
- Provides more placement options without over-constraining
- Maintains musical/logarithmic frequency spacing
- Avoids problematic boundary frequencies

### 3. Frequency Deduplication System

Prevents multiple filters at similar frequencies:

**Collision Detection**:
- Uses 1/10 octave tolerance: `|log2(f1/f2)| < 0.1`
- Tracks used frequencies in optimization loop
- Skips candidates too close to existing filters

**Frequency Selection Priority**:
1. Most effective error reduction
2. Measured frequencies preferred over interpolated
3. Avoids clustering in narrow frequency bands

### 4. Adaptive High-Pass Filter

Automatically places high-pass filter based on usable range:

**Placement Rule**: `f_hp = f_low / 2.0`
- Example: If f_low = 40 Hz, HP placed at 20 Hz
- **Q Factor**: Fixed at 0.5 (gentle slope)
- **Purpose**: Protects speakers from excessive low-frequency energy

**Filter Count Adjustment**: 
- If high-pass enabled: Use `filter_count - 1` PEQ filters + 1 HP
- API automatically adjusts count when HP requested

### 6. Brute-Force Optimization Core

Exhaustive search for optimal filter parameters:

**Search Space**:
- **Frequencies**: 58 intelligent candidates
- **Q Factors**: [0.5, 0.675, 0.9, 1.2, 1.65, 2.2, 3.0, 4.1, 5.5, 7.4, 10.0]
- **Gains**: [-10, -8, -6, -5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3] dB

**Process Per Filter**:
1. Try all frequency × Q × gain combinations
2. Calculate weighted error for each combination
3. Select combination with lowest error
4. Apply filter and update response
5. Repeat for next filter

**Termination**: 
- Continues until all requested filters placed
- Each filter guaranteed to improve (or maintain) response

## Parameter Reference

### Target Curves

#### `flat`
- **Description**: Perfectly flat response (0 dB all frequencies)
- **Use Case**: Studio monitoring, reference systems
- **Weighting**: None (uniform error weighting)

#### `weighted_flat` (Recommended)
- **Description**: Flat response with frequency weighting
- **Use Case**: High-quality listening rooms, audiophile systems
- **Weighting**: Frequency-dependent (emphasizes critical ranges)
- **Curve Points**:
  - 20-100 Hz: Weight [1.0, 0.3-0.6] (bass important, but boosts less critical)
  - 200-500 Hz: Weight [0.9, 0.6-0.7] (midrange critical)
  - 5-10 kHz: Weight 0.1-0.4 (high frequencies less critical)

#### `harman`
- **Description**: Research-based preferred room response
- **Use Case**: Consumer audio, home theaters
- **Characteristics**: 
  - Slight high-frequency roll-off (-1 to -2 dB above 10 kHz)
  - Based on Harman research on preferred room curves
  - Balances accuracy with listening preference

### Optimizer Presets

#### `default` (Recommended)
```json
{
  "qmax": 10.0,
  "mindb": -10.0,
  "maxdb": 3.0,
  "add_highpass": true,
  "acceptable_error": 1.0
}
```
- **Use Case**: General-purpose room correction
- **Characteristics**: Balanced Q factors, moderate correction range
- **Typical Results**: 6-12 dB improvement, 10-15 filters

#### `smooth`
```json
{
  "qmax": 5.0,
  "mindb": -8.0,
  "maxdb": 2.0,
  "add_highpass": true,
  "acceptable_error": 1.0
}
```
- **Use Case**: Gentle correction, vintage systems
- **Characteristics**: Lower Q (broader corrections), reduced gain range
- **Benefits**: Less ringing, more forgiving of measurement errors

#### `aggressive`
```json
{
  "qmax": 15.0,
  "mindb": -15.0,
  "maxdb": 5.0,
  "add_highpass": true,
  "acceptable_error": 1.0
}
```
- **Use Case**: Severe room problems, high-resolution systems
- **Characteristics**: Higher Q (narrower corrections), extended gain range
- **Caution**: May cause ringing, requires accurate measurements

#### Acceptable Error Parameter

**`acceptable_error`** (default: 1.0 dB):
- **Purpose**: Defines the error threshold below which corrections are not considered significant
- **Algorithm**: `adjusted_error = max(0, |raw_error| - acceptable_error)` with sign preservation
- **Effect**: 
  - Errors ≤ 1.0 dB are treated as 0 (no correction needed)
  - Error of +2.5 dB becomes +1.5 dB in optimization calculations  
  - Error of -0.5 dB becomes 0 dB (within acceptable range)
- **Benefits**:
  - Prevents over-correction of minor deviations
  - Focuses optimization on significant problems
  - Reduces unnecessary filter usage for small errors
- **Tuning**:
  - **Lower values** (0.5 dB): More aggressive correction, targets smaller errors
  - **Higher values** (2.0 dB): More forgiving, only corrects major problems

### Filter Count Guidelines

**Recommended Ranges**:
- **Small rooms**: 6-10 filters
- **Medium rooms**: 8-12 filters  
- **Large rooms**: 10-16 filters
- **Severe problems**: 12-20 filters

**Considerations**:
- More filters = better correction but increased complexity
- Each filter adds ~0.1-0.3 dB improvement beyond first 8-10
- High-pass filter counts toward total but provides different benefits

## Performance Metrics

### Typical Results
- **Error Reduction**: 6-12 dB RMS improvement
- **Processing Time**: 3-8 seconds for 8-12 filters
- **Frequency Coverage**: 20-17000 Hz typical usable range

### Performance Factors

**Measurement Quality**:
- **Good measurements**: 8-12 dB improvement possible
- **Poor measurements**: 3-6 dB improvement typical
- **Critical factors**: Microphone calibration, noise floor, averaging

**Room Characteristics**:
- **Treated rooms**: Better correction potential (10-15 dB)
- **Untreated rooms**: Moderate potential (6-10 dB)
- **Small rooms**: More correction needed and achievable

**Filter Count vs. Improvement**:
- Filters 1-4: 1-3 dB each (major corrections)
- Filters 5-8: 0.5-1.5 dB each (fine-tuning)
- Filters 9+: 0.1-0.8 dB each (diminishing returns)

## API Integration

### Streaming Optimization

The REST API provides real-time streaming of optimization progress:

```bash
# Start streaming optimization
curl -X POST http://localhost:10315/eq/optimize/start \
  -H "Content-Type: application/json" \
  -d '{
    "frequencies": [20, 25, 31.5, ...],
    "magnitudes": [-5.2, -3.1, -2.8, ...],
    "target_curve": "weighted_flat",
    "optimizer_preset": "default", 
    "filter_count": 8,
    "add_highpass": true
  }'
```

**Response Format** (Server-Sent Events):
```
data: {"type": "started", "message": "Starting optimization..."}

data: {"type": "initialization", "usable_range": {"f_low": 25.1, "f_high": 16892.4, "candidates": 58}}

data: {"type": "filter_added", "filter": {"frequency": 2500.0, "q": 2.8, "gain_db": -3.8}, "progress": 12.5}

data: {"type": "completed", "result": {"success": true, "improvement_db": 8.3, "filters": [...]}]
```

### High-Pass Integration

When `add_highpass: true`:
- Rust optimizer automatically adds adaptive HP filter
- Filter count adjusted: `effective_peq_count = requested_count - 1`
- HP placement: `f_hp = detected_f_low / 2.0`

## Advanced Topics

### Error Calculation Algorithm

**Weighted RMS Error with Acceptable Threshold**:
```
adjusted_error[i] = {
  0                           if |raw_error[i]| ≤ acceptable_error
  raw_error[i] - acceptable_error  if raw_error[i] > acceptable_error  
  raw_error[i] + acceptable_error  if raw_error[i] < -acceptable_error
}

final_error = sqrt(Σ(w_freq[i] * w_dir[i] * adjusted_error[i]²) / N)
```

**Components**:
- `raw_error[i] = measured[i] - target[i]`
- `acceptable_error`: Threshold below which errors are not penalized (default: 1.0 dB)
- `w_freq[i]`: Frequency-dependent weight
- `w_dir[i]`: Direction-dependent weight (boost vs cut)
- `adjusted_error[i]`: Error after applying acceptable threshold

**Example with acceptable_error = 1.0 dB**:
- Raw error +2.5 dB → Adjusted error +1.5 dB (penalized for excess)
- Raw error +0.8 dB → Adjusted error 0.0 dB (within acceptable range)  
- Raw error -2.0 dB → Adjusted error -1.0 dB (penalized for excess cut)

### Filter Response Calculation

Uses cascaded biquad filters with frequency response:
```
H(f) = Π(H_i(f)) for all filters i
```

Each biquad calculated at measurement frequencies for optimization.

### Convergence Behavior

**Typical Pattern**:
1. **Filters 1-3**: Large error reduction (2-4 dB each)
2. **Filters 4-6**: Moderate reduction (1-2 dB each) 
3. **Filters 7+**: Fine corrections (0.2-0.8 dB each)

**Convergence Indicators**:
- Error reduction per filter decreases
- Filter placement spreads across frequency range
- Q factors may increase (narrower corrections)

## Troubleshooting

### Common Issues

**"No improvement found" messages**:
- **Cause**: All frequency candidates already used or provide no benefit
- **Solution**: Reduce filter count or improve measurement quality

**Very high Q filters (Q > 8)**:
- **Cause**: Narrow room resonances requiring precise correction  
- **Risk**: May cause audible ringing
- **Solution**: Use "smooth" preset or verify measurement accuracy

**Limited frequency range (f_low > 50 Hz or f_high < 10 kHz)**:
- **Cause**: Poor room response or measurement issues
- **Solution**: Check microphone placement, measurement noise floor

**Processing time > 15 seconds**:
- **Cause**: Large filter count or complex optimization space
- **Normal**: Up to 20 filters can take 10-20 seconds
- **Solution**: Reduce filter count for faster results

### Optimization Tips

1. **Start Conservative**: Use 6-8 filters with "default" preset
2. **Verify Measurements**: Good measurements are critical for good results  
3. **Monitor Usable Range**: Should cover at least 40-12000 Hz
4. **Check Filter Distribution**: Should spread across frequency range
5. **Validate Results**: Listen test with generated filters

## Technical Specifications

- **Binary**: Rust executable, ~2-5 MB
- **Memory Usage**: ~10-50 MB during optimization  
- **CPU Usage**: Single-threaded, CPU-intensive during optimization
- **I/O**: Streams JSON output line-by-line
- **Dependencies**: None (statically linked Rust binary)
- **Platforms**: Linux (primary), Windows/macOS (cross-compilation possible)

## Version History

### v0.6.0 (Current Rust Optimizer)
- Brute-force optimization algorithm
- Frequency deduplication and interpolation
- Adaptive high-pass filter placement
- Real-time streaming output

### v0.7.0 (Current REST API)  
- Integrated Rust optimizer backend
- Server-Sent Events streaming
- Automatic high-pass filter count adjustment
- Enhanced error handling and validation
- Comprehensive documentation
