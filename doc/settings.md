# RoomEQ Settings: Target Curves and Optimizer Presets

This document provides detailed information about the target curves and optimizer presets available in the RoomEQ system, including their attributes, use cases, and technical specifications.

## Target Curves

Target curves define the desired frequency response that the EQ optimization system will try to achieve. Each curve consists of frequency-gain pairs with optional weighting parameters that control how aggressively the optimizer corrects deviations from the target.

### Target Curve Attributes

Each target curve has the following attributes:

- **`key`**: Unique identifier for the curve (e.g., "room_only", "flat")
- **`name`**: Human-readable display name
- **`description`**: Detailed explanation of the curve's purpose and characteristics  
- **`expert`**: Boolean flag indicating complexity level
  - `false`: Beginner-friendly curves suitable for general use
  - `true`: Advanced curves requiring acoustic knowledge
- **`curve`**: Array of control points defining the frequency response
  - Format: `[frequency_hz, target_db, weight_parameters]`
  - Weight parameters can be:
    - Single number: `0.5` (equal positive/negative weighting)
    - Tuple: `(pos_weight, neg_weight)` (different weights for corrections above/below target)
    - Triple: `(pos_weight, neg_weight, cutoff)` (with frequency cutoff)

### Available Target Curves

#### 1. Room Correction Only (`room_only`)
- **Expert Level**: Beginner (`expert: false`)
- **Purpose**: Focus correction exclusively on room acoustics issues
- **Frequency Range**: 20Hz - 250Hz
- **Use Case**: Correct room modes and standing waves without affecting speaker characteristics
- **Technical Details**:
  ```
  [20Hz, 0dB, (1.0, 0.5)]    # Strong low-frequency correction
  [250Hz, 0dB]               # Flat response above room modes
  ```

#### 2. Flat Response (`flat`) 
- **Expert Level**: Beginner (`expert: false`)
- **Purpose**: Achieve perfectly flat frequency response
- **Frequency Range**: 20Hz - 25kHz  
- **Use Case**: Studio monitoring, reference listening
- **Technical Details**:
  ```
  [20Hz, 0dB]     # Flat from subsonic
  [25kHz, 0dB]    # Flat to ultrasonic
  ```

#### 3. Falling Slope (`falling_slope`)
- **Expert Level**: Beginner (`expert: false`) 
- **Purpose**: Natural-sounding high-frequency roll-off
- **Frequency Range**: 20Hz - 25kHz with -6dB at 25kHz
- **Use Case**: Warmer, more relaxed sound for casual listening
- **Technical Details**:
  ```
  [20Hz, 0dB, (1.0, 0.3)]      # Room correction focus
  [100Hz, 0dB, (1.0, 0.6)]     # Moderate bass correction
  [200Hz, 0dB, (0.9, 0.7)]     # Balanced midrange
  [500Hz, 0dB, (0.9, 0.5)]     # Gentle mid correction
  [1kHz, 0dB, (0.5, 0.3)]      # Reference frequency
  [10kHz, -3dB, 0.1]           # Gentle treble roll-off
  [25kHz, -6dB]                # Extended roll-off
  ```

#### 4. Weighted Flat (`weighted_flat`)
- **Expert Level**: Advanced (`expert: true`)
- **Purpose**: Flat response with frequency-dependent correction priorities
- **Frequency Range**: 20Hz - 25kHz
- **Use Case**: Precise tuning with acoustic expertise
- **Technical Details**:
  ```
  [20Hz, 0dB, (1.0, 0.3)]      # Strong bass, gentle cuts
  [100Hz, 0dB, (1.0, 0.6)]     # Room mode focus
  [200Hz, 0dB, (0.9, 0.7)]     # Balanced lower mids
  [500Hz, 0dB, 0.6]            # Moderate mid correction
  [5kHz, 0dB, 0.4]             # Light treble correction
  [10kHz, 0dB, 0.1]            # Minimal high-freq changes
  [25kHz, 0dB]                 # Natural treble extension
  ```

#### 5. Harman Curve (`harman`)
- **Expert Level**: Advanced (`expert: true`)
- **Purpose**: Research-based target curve from Harman International
- **Frequency Range**: 20Hz - 20kHz with bass shelf and treble roll-off
- **Use Case**: Consumer audio preference based on listening tests
- **Technical Details**:
  ```
  [20Hz, -2dB, (0.8, 0.4)]     # Slight bass reduction
  [100Hz, 0dB, (1.0, 0.7)]     # Neutral bass
  [200Hz, 0dB, (1.0, 0.8)]     # Clear lower mids
  [500Hz, 0dB, (0.9, 0.7)]     # Balanced mids
  [1kHz, 0dB, (0.7, 0.5)]      # Reference point
  [3kHz, -1dB, (0.6, 0.4)]     # Slight presence dip
  [10kHz, -3dB, (0.3, 0.2)]    # Gentle treble roll-off
  [20kHz, -5dB, (0.1, 0.1)]    # Extended roll-off
  ```

#### 6. Vocal Presence (`vocal_presence`)
- **Expert Level**: Advanced (`expert: true`)
- **Purpose**: Enhanced midrange for vocal clarity and intelligibility
- **Frequency Range**: 20Hz - 20kHz with midrange emphasis
- **Use Case**: Dialogue, podcasts, vocal-centric music
- **Technical Details**:
  ```
  [20Hz, -2dB, (0.5, 0.3)]     # Reduced low-end rumble
  [100Hz, -1dB, (0.7, 0.5)]    # Clean bass
  [200Hz, 0dB, (0.9, 0.7)]     # Neutral lower mids
  [500Hz, 0dB, (1.0, 0.8)]     # Foundation for vocals
  [1kHz, +1dB, (1.0, 0.9)]     # Vocal clarity boost
  [3kHz, +2dB, (1.0, 0.8)]     # Presence enhancement
  [5kHz, 0dB, (0.8, 0.6)]      # Controlled upper mids
  [10kHz, -2dB, (0.4, 0.3)]    # Reduced sibilance
  [20kHz, -4dB, (0.2, 0.1)]    # Smooth treble
  ```

### Weight Parameter Details

The weight parameters control optimization behavior:

- **Single Value** (e.g., `0.5`): Equal weighting for boosts and cuts
- **Positive Weight**: Strength of corrections when signal is below target (boost needed)
- **Negative Weight**: Strength of corrections when signal is above target (cut needed)  
- **Cutoff Parameter**: Frequency cutoff for weight application

**Examples**:
- `(1.0, 0.3)`: Strong boosts, gentle cuts - good for bass extension
- `(0.5, 1.0)`: Gentle boosts, strong cuts - good for taming peaks
- `0.1`: Very light correction in both directions - preserves natural response

## Optimizer Presets

Optimizer presets control how aggressively and precisely the EQ optimization algorithm applies corrections. Each preset defines limits and behavior for the optimization process.

### Optimizer Preset Attributes

- **`key`**: Unique identifier for the preset
- **`name`**: Human-readable display name
- **`description`**: Explanation of the preset's characteristics
- **`qmax`**: Maximum Q factor (filter sharpness) allowed
  - Lower Q = wider, gentler filters
  - Higher Q = narrower, more precise filters
- **`mindb`**: Maximum cut allowed (negative dB value)
- **`maxdb`**: Maximum boost allowed (positive dB value)
- **`add_highpass`**: Whether to automatically add high-pass filter

### Available Optimizer Presets

#### 1. Default (`default`)
- **Purpose**: Balanced optimization suitable for most systems
- **Parameters**:
  - `qmax`: 10 (moderate precision)
  - `mindb`: -10dB (moderate cuts)
  - `maxdb`: +3dB (conservative boosts)
  - `add_highpass`: true (subsonic protection)
- **Use Case**: General-purpose room correction

#### 2. Very Smooth (`verysmooth`)
- **Purpose**: Extremely gentle correction preserving natural sound
- **Parameters**:
  - `qmax`: 2 (very wide filters)
  - `mindb`: -10dB (moderate cuts)
  - `maxdb`: +3dB (conservative boosts)
  - `add_highpass`: true (subsonic protection)
- **Use Case**: High-end systems where natural timbre is critical

#### 3. Smooth (`smooth`)
- **Purpose**: Moderate correction with musical results
- **Parameters**:
  - `qmax`: 5 (medium precision)
  - `mindb`: -10dB (moderate cuts)
  - `maxdb`: +3dB (conservative boosts)
  - `add_highpass`: true (subsonic protection)
- **Use Case**: Hi-fi systems, critical listening

#### 4. Aggressive (`aggressive`)
- **Purpose**: Strong correction for problematic rooms
- **Parameters**:
  - `qmax`: 20 (high precision)
  - `mindb`: -20dB (strong cuts)
  - `maxdb`: +6dB (significant boosts)
  - `add_highpass`: true (subsonic protection)
- **Use Case**: Challenging acoustics, car audio, PA systems

#### 5. Precise (`precise`)
- **Purpose**: High-precision correction for detailed tuning
- **Parameters**:
  - `qmax`: 15 (high precision)
  - `mindb`: -15dB (strong cuts)
  - `maxdb`: +5dB (moderate boosts)
  - `add_highpass`: false (no automatic high-pass)
- **Use Case**: Professional applications, measurement microphones

## Q Factor Guide

Understanding Q factor is crucial for choosing the right optimizer preset:

| Q Factor | Filter Width | Characteristics | Use Case |
|----------|--------------|----------------|-----------|
| 0.5-1.0  | Very Wide    | Natural, musical | Broad tonal adjustments |
| 2-5      | Wide         | Smooth correction | Room mode correction |
| 5-10     | Medium       | Balanced precision | General purpose |
| 10-15    | Narrow       | Precise targeting | Problem frequencies |
| 15+      | Very Narrow  | Surgical precision | Specific resonances |

## Recommended Combinations

### For Beginners
- **Target**: `room_only` + **Optimizer**: `smooth`
- **Purpose**: Focus on room acoustics with gentle correction
- **Result**: Improved bass response without affecting speaker character

### For Balanced Results  
- **Target**: `falling_slope` + **Optimizer**: `default`
- **Purpose**: Natural-sounding full-range correction
- **Result**: Warmer, more relaxed sound suitable for extended listening

### For Critical Listening
- **Target**: `weighted_flat` + **Optimizer**: `precise` 
- **Purpose**: Accurate reference monitoring
- **Result**: Detailed, uncolored sound for mixing and mastering

### For Vocal Content
- **Target**: `vocal_presence` + **Optimizer**: `smooth`
- **Purpose**: Enhanced speech intelligibility
- **Result**: Clear, articulate vocals with reduced listening fatigue

## Advanced Usage Notes

### Custom Weighting Strategy
- Use higher positive weights for frequencies you want to boost (bass extension)
- Use higher negative weights for frequencies you want to control (room peaks)
- Use low weights (0.1-0.3) in the 10kHz+ range to preserve air and detail

### High-Pass Filter Considerations
- **Enable** (`add_highpass: true`) for:
  - Bookshelf speakers (protects from over-excursion)
  - Car audio (removes road noise)
  - PA systems (feedback prevention)
- **Disable** (`add_highpass: false`) for:
  - Full-range speakers with good low-end extension
  - Subwoofer integration systems
  - Measurement applications

### Expert vs Beginner Classification
- **Beginner** curves (`expert: false`) are safer and more forgiving
- **Expert** curves (`expert: true`) require acoustic knowledge and careful system matching
- Start with beginner curves and progress to expert curves as you gain experience

## API Integration

All target curves and optimizer presets are accessible via the REST API:

- **GET /eq/presets/targets** - List all target curves with metadata
- **GET /eq/presets/optimizers** - List all optimizer presets with parameters  
- **POST /eq/optimize/start** - Start optimization with specified curve and preset

Each API response includes the `key` field for programmatic access and all attributes documented above.

## Quick Reference Tables

### Target Curves Summary

| Key | Name | Expert | Frequency Range | Purpose |
|-----|------|--------|----------------|---------|
| `room_only` | Room Correction Only | No | 20-250Hz | Room modes only |
| `flat` | Flat Response | No | 20Hz-25kHz | Studio reference |
| `falling_slope` | Falling Slope | No | 20Hz-25kHz (-6dB) | Warm, relaxed |
| `weighted_flat` | Weighted Flat | Yes | 20Hz-25kHz | Precise flat |
| `harman` | Harman Curve | Yes | 20Hz-20kHz | Consumer preference |
| `vocal_presence` | Vocal Presence | Yes | 20Hz-20kHz (+2dB mids) | Speech clarity |

### Optimizer Presets Summary

| Key | Name | Q Max | Min dB | Max dB | High-Pass | Use Case |
|-----|------|-------|--------|--------|-----------|----------|
| `default` | Default | 10 | -10 | +3 | Yes | General purpose |
| `verysmooth` | Very Smooth | 2 | -10 | +3 | Yes | Natural sound |
| `smooth` | Smooth | 5 | -10 | +3 | Yes | Hi-fi systems |
| `aggressive` | Aggressive | 20 | -20 | +6 | Yes | Problem rooms |
| `precise` | Precise | 15 | -15 | +5 | No | Professional |

### Beginner-Friendly Quick Start

1. **First Time Users**: `room_only` + `smooth`
2. **Warm Sound**: `falling_slope` + `default`
3. **Accurate Reference**: `flat` + `default`
4. **Problem Room**: `room_only` + `aggressive`

### Expert Combinations

1. **Studio Monitoring**: `weighted_flat` + `precise`
2. **Consumer Tuning**: `harman` + `smooth`
3. **Broadcast Audio**: `vocal_presence` + `default`
4. **Measurement Work**: `flat` + `precise`
