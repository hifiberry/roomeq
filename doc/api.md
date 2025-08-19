# RoomEQ Audio Processing API Documentation

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Authentication](#authentication)
- [Endpoints Reference](#endpoints-reference)
  - [Information Endpoints](#information-endpoints)
    - [GET `/`](#get-)
    - [GET `/version`](#get-version)
  - [Microphone Detection](#microphone-detection)
    - [GET `/microphones`](#get-microphones)
    - [GET `/microphones/raw`](#get-microphonesraw)
  - [Audio Device Information](#audio-device-information)
    - [GET `/audio/inputs`](#get-audioinputs)
    - [GET `/audio/cards`](#get-audiocards)
  - [SPL Measurement](#spl-measurement)
    - [GET `/spl/measure`](#get-splmeasure)
  - [Signal Generation](#signal-generation)
    - [POST `/audio/noise/start`](#post-audionoisestart)
    - [POST `/audio/noise/keep-playing`](#post-audionoisekeep-playing)
    - [POST `/audio/noise/stop`](#post-audionoisestop)
    - [GET `/audio/noise/status`](#get-audionoisestatus)
  - [Sine Sweep Generation](#sine-sweep-generation)
    - [POST `/audio/sweep/start`](#post-audiosweepstart)
- [FFT Analysis](#fft-analysis)
  - [POST `/audio/analyze/fft`](#audioanalyzefft-post)
  - [FFT Analysis Technical Details](#fft-analysis-technical-details)
    - [Power Spectral Density (PSD) Calculation](#power-spectral-density-psd-calculation)
    - [Logarithmic Frequency Summarization](#logarithmic-frequency-summarization)
    - [Psychoacoustic Smoothing](#psychoacoustic-smoothing)
  - [POST `/audio/analyze/fft-recording/<recording_id>`](#audioanalyzefft-recordingrecording_id-post)
- [EQ Optimization](#eq-optimization)
  - [Target Curves and Optimizer Presets](#target-curves-and-optimizer-presets)
    - [GET `/eq/presets/targets`](#get-eqpresetstargets)
    - [GET `/eq/presets/optimizers`](#get-eqpresetsoptimizers)
  - [EQ Optimization Process](#eq-optimization-process)
    - [POST `/eq/optimize`](#post-eqoptimize)
  - [Legacy Async API Endpoints (Deprecated)](#legacy-async-api-endpoints-deprecated)
    - [GET `/eq/optimize/status/<optimization_id>`](#get-eqoptimizestatusoptimization_id)
    - [POST `/eq/optimize/cancel/<optimization_id>`](#post-eqoptimizecanceloptimization_id)
    - [GET `/eq/optimize/result/<optimization_id>`](#get-eqoptimizeresultoptimization_id)
  - [EQ Optimization Usage Examples](#eq-optimization-usage-examples)
    - [Complete Room Correction Workflow](#complete-room-correction-workflow)
    - [Quick Optimization from Existing FFT Data](#quick-optimization-from-existing-fft-data)
    - [Monitor Multiple Optimizations](#monitor-multiple-optimizations)
- [Audio Recording](#audio-recording)
  - [Background Recording to WAV Files](#background-recording-to-wav-files)
    - [POST `/audio/record/start`](#post-audiorecordstart)
    - [GET `/audio/record/status/<recording_id>`](#get-audiorecordstatusrecording_id)
    - [GET `/audio/record/list`](#get-audiorecordlist)
    - [GET `/audio/record/download/<recording_id>`](#get-audiorecorddownloadrecording_id)
    - [DELETE `/audio/record/delete/<recording_id>`](#delete-audiorecorddeleterecording_id)
    - [DELETE `/audio/record/delete-file/<filename>`](#delete-audiorecorddelete-filefilename)
- [Usage Examples](#usage-examples)
  - [Basic SPL Measurement](#basic-spl-measurement)
  - [Noise Generation with Keep-Alive](#noise-generation-with-keep-alive)
  - [Sine Sweep Generation](#sine-sweep-generation-1)
  - [Audio Recording Workflow](#audio-recording-workflow)
  - [FFT Analysis Examples](#fft-analysis-examples)

## Overview

The RoomEQ Audio Processing API provides a comprehensive REST interface for microphone detection, sound pressure level (SPL) measurement, audio signal generation, recording, FFT analysis, and **automatic room EQ optimization** functionality. This API is designed for acoustic measurement systems, room correction, and audio testing applications with real-time progress reporting.

**Base URL:** `http://localhost:10315`  
**API Version:** 0.6.0  
**Framework:** Flask with CORS support
**Documentation:** Available at the root endpoint `/`

## Features

- **Microphone Detection**: Automatic detection of USB and built-in microphones with sensitivity and gain information
- **SPL Measurement**: Accurate sound pressure level measurement with calibrated microphones
- **Signal Generation**: White noise and logarithmic sine sweep generation with keep-alive functionality
- **Multiple Sweep Support**: Generate consecutive sine sweeps for acoustic averaging
- **Audio Recording**: Background recording to WAV files with secure file management
- **FFT Analysis**: Comprehensive spectral analysis with windowing functions, normalization, and logarithmic frequency summarization
- **EQ Optimization**: Automatic room EQ optimization with multiple target curves and real-time progress reporting
- **Biquad Filter Generation**: Generate parametric EQ filters with complete coefficient sets
- **Advanced Optimization**: Rust-based high-performance optimization with frequency response calculation
- **Real-time Control**: Start, stop, extend, and monitor operations through REST endpoints
- **Cross-Origin Support**: CORS enabled for web application integration

## Authentication

Currently, no authentication is required for API access.

## Endpoints Reference

### Information Endpoints

#### GET `/`
Get API information and endpoint overview.

**Response:**
```json
{
  "message": "RoomEQ Audio Processing API",
  "version": "0.3.0",
  "framework": "Flask",
  "description": "REST API for microphone detection, SPL measurement, audio signal generation, recording, and FFT analysis for acoustic measurements and room equalization",
  "endpoints": {
    "info": {"/": "API information", "/version": "Version details"},
    "microphones": {"/microphones": "Detect microphones", "/microphones/raw": "Raw detection"},
    "audio_devices": {"/audio/inputs": "Input cards", "/audio/cards": "All cards"},
    "measurements": {"/spl/measure": "SPL measurement"},
    "signal_generation": {
      "/audio/noise/start": "White noise playback",
      "/audio/noise/keep-playing": "Extend playbook", 
      "/audio/noise/stop": "Stop playback",
      "/audio/noise/status": "Playbook status",
      "/audio/sweep/start": "Sine sweep generation"
    },
    "fft_analysis": {
      "/audio/analyze/fft": "FFT analysis of WAV files",
      "/audio/analyze/fft-recording/<id>": "FFT analysis of recordings"
    },
    "recording": {
      "/audio/record/start": "Start recording",
      "/audio/record/status/<id>": "Recording status",
      "/audio/record/list": "List recordings",
      "/audio/record/download/<id>": "Download recording",
      "/audio/record/delete/<id>": "Delete recording"
    }
  }
}
```

#### GET `/version`
Get detailed version information.

**Example Request:**
```bash
curl -X GET http://localhost:10315/version
```

**Response:**
```json
{
  "version": "0.6.0",
  "api_name": "RoomEQ Audio Processing API",
  "features": [
    "Microphone detection with sensitivity and gain",
    "SPL measurement",
    "FFT analysis with windowing, normalization, and logarithmic frequency summarization",
    "Audio recording with automatic cleanup",
    "Sine sweep generation",
    "White/pink noise generation",
    "Automatic room EQ optimization with multiple target curves",
    "Real-time optimization progress reporting",
    "Parametric EQ filter generation (biquad coefficients)",
    "Multiple optimizer presets with different smoothing characteristics",
    "Rust-based high-performance optimization algorithms"
  ],
  "server_info": {
    "python_version": "3.11.2",
    "flask_version": "2.x",
    "threading": "Multi-threaded request handling",
    "audio_backend": "ALSA with arecord fallback for compatibility",
    "optimization_backend": "Rust optimizer with least squares curve fitting and real-time frequency response"
  }
}
```

### Microphone Detection

#### GET `/microphones`
Get detected microphones with detailed properties.

**Example Request:**
```bash
curl -X GET http://localhost:10315/microphones
```

**Response:**
```json
[
  {
    "card_index": 1,
    "device_name": "Dayton UMM6",
    "sensitivity": 137.5,
    "sensitivity_str": "137.5",
    "gain_db": 20.0
  }
]
```

**Response Fields:**
- `card_index`: ALSA card index for the microphone
- `device_name`: Human-readable device name
- `sensitivity`: Microphone sensitivity in dB SPL (numeric)
- `sensitivity_str`: Microphone sensitivity as string
- `gain_db`: Current microphone gain setting in dB

#### GET `/microphones/raw`
Get microphones in raw format (bash script compatible).

**Example Request:**
```bash
curl -X GET http://localhost:10315/microphones/raw
```

**Response:**
```json
{
  "microphones": ["1:Dayton UMM6:137.5"]
}
```

### Audio Device Information

#### GET `/audio/inputs`
Get available audio input cards.

**Response:**
```json
{
  "input_cards": [1],
  "count": 1
}
```

#### GET `/audio/cards`
Get all available ALSA audio cards.

**Response:**
```json
{
  "cards": [
    "0 [sndrpihifiberry]: RPi-simple - snd_rpi_hifiberry_dac",
    "1 [UMM6           ]: USB-Audio - UMM-6"
  ],
  "count": 2
}
```

### SPL Measurement

#### GET `/spl/measure`
Measure sound pressure level using specified or auto-detected microphone.

**Query Parameters:**
- `device` (optional): ALSA device name (e.g., "hw:1,0"). Auto-detects if not specified
- `duration` (optional): Measurement duration in seconds (0.1-10.0, default: 1.0)

**Example Requests:**
```bash
# Basic SPL measurement with auto-detection
curl -X GET http://localhost:10315/spl/measure

# Specific device and duration
curl -X GET "http://localhost:10315/spl/measure?device=hw:1,0&duration=2.0"

# Long measurement for stable reading
curl -X GET "http://localhost:10315/spl/measure?duration=5.0"
```

**Response:**
```json
{
  "spl_db": 75.8,
  "rms_db_fs": -41.7,
  "device": "hw:1,0",
  "duration": 2.0,
  "microphone": {
    "sensitivity": 137.5,
    "gain_db": 20.0,
    "effective_sensitivity": 117.5
  },
  "timestamp": 1692020345.123,
  "success": true
}
```

**Response Fields:**
- `spl_db`: Measured sound pressure level in dB SPL
- `rms_db_fs`: RMS level relative to full scale in dB
- `device`: ALSA device used for measurement
- `duration`: Actual measurement duration
- `microphone`: Microphone calibration information
- `timestamp`: Unix timestamp of measurement
- `success`: Measurement success status

### Signal Generation

#### POST `/audio/noise/start`
Start white noise playback with automatic timeout.

**Query Parameters:**
- `duration` (optional): Initial playback duration in seconds (1.0-30.0, default: 3.0)
- `amplitude` (optional): Amplitude level (0.0-1.0, default: 0.5)
- `device` (optional): Output device (e.g., "hw:0,0"). Uses default if not specified

**Example Requests:**
```bash
# Basic noise generation
curl -X POST http://localhost:10315/audio/noise/start

# Custom duration and amplitude
curl -X POST "http://localhost:10315/audio/noise/start?duration=5&amplitude=0.3"

# Specific output device
curl -X POST "http://localhost:10315/audio/noise/start?duration=10&amplitude=0.4&device=hw:0,0"
```

**Response:**
```json
{
  "status": "started",
  "duration": 5.0,
  "amplitude": 0.3,
  "device": "hw:0,0",
  "stop_time": "2025-08-18T15:30:45.123456",
  "message": "Noise playback started for 5.0 seconds"
}
```

#### POST `/audio/noise/keep-playing`
Extend current noise playback duration (keep-alive mechanism).

**Query Parameters:**
- `duration` (optional): Additional duration in seconds (1.0-30.0, default: 3.0)

**Example Requests:**
```bash
# Extend by default 3 seconds
curl -X POST http://localhost:10315/audio/noise/keep-playing

# Extend by specific duration
curl -X POST "http://localhost:10315/audio/noise/keep-playing?duration=5"
```

**Response:**
```json
{
  "status": "extended",
  "duration": 3.0,
  "new_stop_time": "2025-08-14T15:30:48.123456",
  "message": "Playback extended by 3.0 seconds"
}
```

**Error Response (no active playback):**
```json
{
  "detail": "No active noise playback to extend"
}
```

#### POST `/audio/noise/stop`
Stop current noise playback immediately.

**Example Request:**
```bash
curl -X POST http://localhost:10315/audio/noise/stop
```

**Response:**
```json
{
  "status": "stopped",
  "message": "Noise playback stopped"
}
```

**Response (no active playback):**
```json
{
  "status": "not_active",
  "message": "No active noise playback to stop"
}
```

#### GET `/audio/noise/status`
Get current noise playback status.

**Example Request:**
```bash
curl -X GET http://localhost:10315/audio/noise/status
```

**Response:**
```json
{
  "active": true,
  "amplitude": 0.5,
  "device": "default",
  "remaining_seconds": 2.3,
  "stop_time": "2025-08-14T15:30:45.123456"
}
```

**Response Fields:**
- `active`: Whether audio is currently playing
- `signal_type`: Type of signal being played ("noise" or "sine_sweep")
- `amplitude`: Current playback amplitude
- `device`: Output device being used
- `remaining_seconds`: Seconds until automatic stop
- `stop_time`: ISO timestamp when playback will stop

For sine sweeps, additional fields are included:
- `start_freq`: Starting frequency in Hz
- `end_freq`: Ending frequency in Hz
- `sweeps`: Number of consecutive sweeps
- `sweep_duration`: Duration per sweep in seconds
- `total_duration`: Total duration of all sweeps
- `compensation_mode`: Amplitude compensation for native sweep generator (`"none" | "inv_sqrt_f" | "sqrt_f"`, default `"sqrt_f"`)
- `generator`: Signal source used (`"native" | "sine_sox"`)

### Sine Sweep Generation

#### POST `/audio/sweep/start`
Start logarithmic sine sweep(s) with optional multiple repeat support.

**Query Parameters:**
- `start_freq` (optional): Starting frequency in Hz (10-22000, default: 20)
- `end_freq` (optional): Ending frequency in Hz (10-22000, default: 20000)
- `duration` (optional): Duration per sweep in seconds (1.0-30.0, default: 5.0)
- `sweeps` (optional): Number of consecutive sweeps (1-10, default: 1)
- `amplitude` (optional): Amplitude level (0.0-1.0, default: 0.5)
- `compensation_mode` (optional): Amplitude compensation envelope for the native generator (`none | inv_sqrt_f | sqrt_f`, default: `sqrt_f`)
- `generator` (optional): Signal generator implementation to use (`native | sine_sox`, default: `native`). When `sine_sox` is used, a temporary WAV is generated via SoX in `/tmp`, loaded, then deleted before playback.
- `device` (optional): Output device (e.g., "hw:0,0"). Uses default if not specified

**Example Requests:**
```bash
# Single sweep - full spectrum, 10 seconds
POST /audio/sweep/start?start_freq=20&end_freq=20000&duration=10&amplitude=0.4

# Multiple sweeps for averaging - 3 consecutive sweeps
POST /audio/sweep/start?start_freq=100&end_freq=8000&duration=5&sweeps=3&amplitude=0.3

# Use SoX-based generator (creates a temp WAV in /tmp, then starts playback)
POST /audio/sweep/start?start_freq=20&end_freq=20000&duration=8&sweeps=2&amplitude=0.3&generator=sine_sox
```

**Response:**
```json
{
  "status": "started",
  "signal_type": "sine_sweep",
  "start_freq": 100.0,
  "end_freq": 8000.0,
  "duration": 5.0,
  "sweeps": 3,
  "total_duration": 15.0,
  "amplitude": 0.3,
    "compensation_mode": "sqrt_f",
    "generator": "native",
  "device": "default",
  "stop_time": "2025-08-15T12:30:45.123456",
  "message": "3 sine sweep(s) started: 100.0 Hz → 8000.0 Hz, 5.0s each (total: 15.0s)"
}
```

**Behavior Notes:**
- When `generator=sine_sox`, the server blocks until SoX has created the temporary WAV file and playback has started, then returns the response. The WAV file is removed after loading.
- The `compensation_mode` parameter applies only to the native generator. The SoX generator uses SoX’s sine sweep synthesis without the internal compensation envelope.

**Use Cases:**
- **Room Response Analysis**: Full spectrum sweeps (20 Hz - 20 kHz)
- **Speaker Testing**: Focused frequency ranges
- **Acoustic Averaging**: Multiple sweeps for noise reduction
- **Automated Measurements**: Scripted measurement sequences

## FFT Analysis

### `/audio/analyze/fft` [POST]
Perform FFT (Fast Fourier Transform) spectral analysis on a WAV file.

**Parameters:**
- `filename` (string, optional): Name of previously recorded file to analyze
- `filepath` (string, optional): Full path to WAV file (external files)
- `window` (string, optional): Window function - "hann", "hamming", "blackman", or "none" (default: "hann")
- `fft_size` (integer, optional): FFT size, must be power of 2 between 64-65536 (auto-calculated if not specified)
- `start_time` (float, optional): Start analysis at this time in seconds (default: 0)
- `start_at` (float, optional): Alternative name for `start_time` - start analysis at this time in seconds
- `duration` (float, optional): Duration to analyze in seconds (default: entire file from start_time)
- `normalize` (float, optional): Frequency in Hz to normalize to 0 dB (all other levels adjusted relative to this frequency)
- `points_per_octave` (integer, optional): Summarize FFT into logarithmic frequency buckets (1-100, enables log frequency summarization)
- `psychoacoustic_smoothing` (float, optional): Apply psychoacoustic smoothing with critical band filtering (0.1-5.0, where 1.0 is normal strength)

**Note:** Must specify either `filename` OR `filepath`, not both. Use either `start_time` OR `start_at`, not both (start_at takes precedence if both are provided).

**Success Response (200):**
```json
{
    "status": "success",
    "file_info": {
        "filename": "recording_20240101_120000.wav",
        "original_metadata": {
            "duration": 10.5,
            "sample_rate": 44100,
            "channels": 1,
            "bit_depth": 16,
            "total_samples": 463050
        },
        "analyzed_duration": 5.0,
        "analyzed_samples": 220500,
        "start_time": 2.0
    },
    "fft_analysis": {
        "fft_size": 8192,
        "window_type": "hann",
        "sample_rate": 44100,
        "frequency_resolution": 5.383,
        "frequencies": [0, 5.383, 10.766, ...],
        "magnitudes": [-80.5, -75.2, -82.1, ...],
        "phases": [0.0, 1.57, -1.23, ...],
        "peak_frequency": 1000.0,
        "peak_magnitude": -20.5,
        "spectral_centroid": 1250.5,
        "frequency_bands": {
            "sub_bass": {"range": "20-60 Hz", "avg_magnitude": -65.2, "peak_frequency": 45.0},
            "bass": {"range": "60-250 Hz", "avg_magnitude": -45.8, "peak_frequency": 120.0},
            "low_midrange": {"range": "250-500 Hz", "avg_magnitude": -35.1, "peak_frequency": 350.0},
            "midrange": {"range": "500-2000 Hz", "avg_magnitude": -25.6, "peak_frequency": 1000.0},
            "upper_midrange": {"range": "2000-4000 Hz", "avg_magnitude": -30.2, "peak_frequency": 2500.0},
            "presence": {"range": "4000-6000 Hz", "avg_magnitude": -40.5, "peak_frequency": 5000.0},
            "brilliance": {"range": "6000-20000 Hz", "avg_magnitude": -50.8, "peak_frequency": 8000.0}
        },
        "normalization": {
            "applied": true,
            "requested_freq": 1000.0,
            "actual_freq": 1000.0,
            "reference_level_db": -20.5
        }
    },
    "timestamp": "2024-01-01T12:00:00"
}
```

**Success Response with Logarithmic Frequency Summarization (when `points_per_octave` is specified):**
```json
{
    "status": "success",
    "file_info": {
        "filename": "recording_20240101_120000.wav",
        "original_metadata": {
            "duration": 10.5,
            "sample_rate": 44100,
            "channels": 1,
            "bit_depth": 16,
            "total_samples": 463050
        },
        "analyzed_duration": 5.0,
        "analyzed_samples": 220500,
        "start_time": 2.0
    },
    "fft_analysis": {
        "fft_size": 8192,
        "window_type": "hann",
        "sample_rate": 44100,
        "frequency_resolution": 5.383,
        "frequencies": [0, 5.383, 10.766, ...],
        "magnitudes": [-80.5, -75.2, -82.1, ...],
        "phases": [0.0, 1.57, -1.23, ...],
        "peak_frequency": 1000.0,
        "peak_magnitude": -20.5,
        "spectral_centroid": 1250.5,
        "log_frequency_summary": {
            "frequencies": [20.0, 23.8, 28.3, 33.7, 40.1, 47.8, 56.8, 67.6, 80.4, 95.6, 113.8, 135.4, ...],
            "magnitudes": [-65.2, -62.8, -58.4, -55.1, -52.3, -48.9, -45.2, -42.1, -38.8, -35.5, -32.2, -28.9, ...],
            "points_per_octave": 16,
            "frequency_range": [20.0, 20000.0],
            "n_octaves": 10.0,
            "n_points": 161,
            "bin_details": [
                {
                    "center_freq": 20.0,
                    "freq_range": [17.8, 22.4],
                    "n_samples": 3,
                    "mean_magnitude": -65.2,
                    "min_magnitude": -67.1,
                    "max_magnitude": -63.5
                },
                {
                    "center_freq": 23.8,
                    "freq_range": [21.2, 26.7],
                    "n_samples": 4,
                    "mean_magnitude": -62.8,
                    "min_magnitude": -64.2,
                    "max_magnitude": -61.1
                }
            ]
        },
        "frequency_bands": {
            "sub_bass": {"range": "20-60 Hz", "avg_magnitude": -65.2, "peak_frequency": 45.0},
            "bass": {"range": "60-250 Hz", "avg_magnitude": -45.8, "peak_frequency": 120.0},
            "low_midrange": {"range": "250-500 Hz", "avg_magnitude": -35.1, "peak_frequency": 350.0},
            "midrange": {"range": "500-2000 Hz", "avg_magnitude": -25.6, "peak_frequency": 1000.0},
            "upper_midrange": {"range": "2000-4000 Hz", "avg_magnitude": -30.2, "peak_frequency": 2500.0},
            "presence": {"range": "4000-6000 Hz", "avg_magnitude": -40.5, "peak_frequency": 5000.0},
            "brilliance": {"range": "6000-20000 Hz", "avg_magnitude": -50.8, "peak_frequency": 8000.0}
        },
        "normalization": {
            "applied": true,
            "requested_freq": 1000.0,
            "actual_freq": 1000.0,
            "reference_level_db": -20.5
        }
    },
    "timestamp": "2024-01-01T12:00:00"
}
```

### FFT Analysis Technical Details

#### Power Spectral Density (PSD) Calculation
The FFT analysis has been enhanced with proper Power Spectral Density calculation to ensure accurate spectral measurements:

- **Window Correction**: Uses Equivalent Noise Bandwidth (ENBW) correction for proper spectral density normalization
- **Single-Sided Spectrum**: Correctly scales for single-sided spectrum representation (doubling all frequencies except DC and Nyquist)
- **Proper dB Conversion**: Uses `10*log10` for power density (not `20*log10` which is for magnitude)
- **Spectral Density Units**: Results are in dB relative to full scale squared per Hz (dB re FS²/Hz)

These improvements eliminate the artificial 3dB/octave roll-off that can occur with simple magnitude-based calculations when using logarithmic frequency averaging. This ensures:
- White noise shows a flat response across all frequencies
- Sine waves show their true amplitude regardless of frequency  
- Spectral density is properly normalized for the window function used

#### Logarithmic Frequency Summarization

When the `points_per_octave` parameter is specified, the FFT analysis includes an additional `log_frequency_summary` section that groups the high-resolution FFT data into logarithmically-spaced frequency buckets. This is particularly useful for acoustic analysis where human hearing and audio systems respond logarithmically to frequency.

**Key Benefits:**
- **Perceptual Relevance**: Matches human auditory perception (logarithmic frequency response)
- **Data Reduction**: Reduces thousands of FFT bins to manageable number of points (typically 50-200)
- **Consistent Resolution**: Same number of points per octave across the entire frequency range
- **Analysis Friendly**: Perfect for room EQ, speaker analysis, and acoustic measurement applications

**Typical Values:**
- `points_per_octave=12`: 121 points from 20Hz to 20kHz (similar to 1/3 octave bands)
- `points_per_octave=16`: 161 points (higher resolution, good for detailed analysis)
- `points_per_octave=24`: 241 points (very high resolution for research applications)

#### Psychoacoustic Smoothing

The FFT analysis supports optional psychoacoustic smoothing that applies frequency-dependent bandwidth filtering to match human auditory perception. This smoothing uses critical bands based on the Bark scale:

**Key Features:**
- **Variable Bandwidth**: Narrow smoothing at low frequencies (25-100 Hz bandwidth), progressively wider at high frequencies (up to 1000+ Hz bandwidth)
- **Perceptual Accuracy**: Follows critical bands of human hearing for more natural frequency response presentation
- **Noise Reduction**: Reduces measurement noise while preserving important spectral features
- **Visual Enhancement**: Creates cleaner, more readable frequency response plots

**Smoothing Factor Values:**
- `0.5`: Light smoothing, preserves most detail
- `1.0`: Standard psychoacoustic smoothing (recommended)
- `1.5`: Moderate smoothing, good for noisy measurements
- `2.0`: Heavy smoothing, removes fine detail but shows overall trends
- `3.0+`: Very heavy smoothing for extremely noisy data

**When to Use:**
- **Room measurements**: Smooths room resonances and reflections for clearer overall response
- **Speaker testing**: Reduces driver breakup and baffle effects while showing overall response shape
- **Noisy environments**: Improves signal-to-noise ratio in measurements with background noise
- **Publication plots**: Creates cleaner graphs for reports and presentations

**Technical Implementation:**
- Uses Gaussian-weighted averaging within frequency-dependent critical bands
- Preserves energy conservation by operating in linear power domain
- Applies Bark scale bandwidth approximation for computational efficiency

**Log Frequency Summary Fields:**
- `frequencies`: Center frequencies of logarithmic bins
- `magnitudes`: Mean magnitude in dB for each frequency bin
- `points_per_octave`: Number of frequency points per octave
- `frequency_range`: Actual frequency range covered [f_min, f_max]
- `n_octaves`: Number of octaves in the analysis
- `n_points`: Total number of frequency points in summary
- `bin_details`: Detailed information about each frequency bin (optional, for debugging)
```

### `/audio/analyze/fft-recording/<recording_id>` [POST]
Perform FFT analysis on a specific recording by ID.

**Path Parameters:**
- `recording_id`: ID of the recording to analyze

**Query Parameters:**
- `window` (string, optional): Window function (default: "hann")
- `fft_size` (integer, optional): FFT size, must be power of 2
- `start_time` (float, optional): Start analysis time in seconds
- `start_at` (float, optional): Alternative name for `start_time` - start analysis time in seconds  
- `duration` (float, optional): Duration to analyze in seconds
- `normalize` (float, optional): Frequency in Hz to normalize to 0 dB
- `points_per_octave` (integer, optional): Summarize FFT into logarithmic frequency buckets (1-100)
- `psychoacoustic_smoothing` (float, optional): Apply psychoacoustic smoothing (0.1-5.0)

**Note:** Use either `start_time` OR `start_at`, not both (start_at takes precedence if both are provided).

**Success Response (200):**
```json
{
    "status": "success",
    "recording_info": {
        "recording_id": "rec_20240101_120000_abc123",
        "filename": "recording_20240101_120000.wav",
        "original_duration": 10.5,
        "original_device": "hw:1,0",
        "original_sample_rate": 44100,
        "timestamp": "2024-01-01T12:00:00"
    },
    "analysis_info": {
        "analyzed_duration": 10.5,
        "analyzed_samples": 463050,
        "start_time": 0
    },
    "fft_analysis": {
        "fft_size": 8192,
        "window_type": "hann",
        "sample_rate": 44100,
        "frequency_resolution": 5.383,
        "frequencies": [0, 5.383, 10.766, ...],
        "magnitudes": [-80.5, -75.2, -82.1, ...],
        "phases": [0.0, 1.57, -1.23, ...],
        "peak_frequency": 1000.0,
        "peak_magnitude": -20.5,
        "spectral_centroid": 1250.5,
        "frequency_bands": {
            "sub_bass": {"range": "20-60 Hz", "avg_magnitude": -65.2, "peak_frequency": 45.0},
            "bass": {"range": "60-250 Hz", "avg_magnitude": -45.8, "peak_frequency": 120.0},
            "low_midrange": {"range": "250-500 Hz", "avg_magnitude": -35.1, "peak_frequency": 350.0},
            "midrange": {"range": "500-2000 Hz", "avg_magnitude": -25.6, "peak_frequency": 1000.0},
            "upper_midrange": {"range": "2000-4000 Hz", "avg_magnitude": -30.2, "peak_frequency": 2500.0},
            "presence": {"range": "4000-6000 Hz", "avg_magnitude": -40.5, "peak_frequency": 5000.0},
            "brilliance": {"range": "6000-20000 Hz", "avg_magnitude": -50.8, "peak_frequency": 8000.0}
        }
    },
    "analysis_timestamp": "2024-01-01T12:05:00"
}
```

**Error Responses:**
- `400 Bad Request`: Invalid parameters or file format
- `404 Not Found`: Recording or file not found
- `500 Internal Server Error`: Analysis processing error

## EQ Optimization

The API provides advanced room EQ optimization functionality with real-time progress reporting. The optimization system uses a high-performance Rust-based optimizer to generate parametric EQ filters that correct room response to match target curves.

### Target Curves and Optimizer Presets

#### GET `/eq/presets/targets`
List available target response curves for optimization.

**Example Request:**
```bash
curl -X GET http://localhost:10315/eq/presets/targets
```

**Response:**
```json
{
  "success": true,
  "count": 6,
  "target_curves": [
    {
      "key": "flat",
      "name": "Flat Response",
      "description": "Perfectly flat frequency response (0 dB across all frequencies)",
      "expert": false,
      "curve": [
        {"frequency": 20.0, "target_db": 0.0, "weight": null},
        {"frequency": 25000.0, "target_db": 0.0, "weight": null}
      ]
    },
    {
      "key": "harman",
      "name": "Harman Target Curve",
      "description": "Research-based preferred room response with gentle high-frequency roll-off",
      "expert": true,
      "curve": [
        {"frequency": 20.0, "target_db": 0.0, "weight": 0.8},
        {"frequency": 100.0, "target_db": 0.0, "weight": 1.0},
        {"frequency": 1000.0, "target_db": 0.0, "weight": 1.0},
        {"frequency": 10000.0, "target_db": -1.0, "weight": 0.5},
        {"frequency": 20000.0, "target_db": -2.0, "weight": 0.3}
      ]
    },
    {
      "key": "falling_slope",
      "name": "Falling Slope",
      "description": "Gentle downward slope for warmer sound",
      "expert": false,
      "curve": [
        {"frequency": 20.0, "target_db": 2.0, "weight": 0.6},
        {"frequency": 100.0, "target_db": 1.0, "weight": 0.8},
        {"frequency": 1000.0, "target_db": 0.0, "weight": 1.0},
        {"frequency": 10000.0, "target_db": -2.0, "weight": 0.6},
        {"frequency": 20000.0, "target_db": -4.0, "weight": 0.3}
      ]
    }
  ]
}
```

**Curve Data Structure:**
Each curve point contains:
- `frequency`: Frequency in Hz
- `target_db`: Target gain/attenuation in dB at this frequency
- `weight`: Optimization weight (null = default, number = custom weight, array = [freq_weight, magnitude_weight])

**Field Descriptions:**
- `key`: Unique identifier for the target curve
- `name`: Human-readable name
- `description`: Detailed description of the curve characteristics
- `expert`: Boolean indicating if this is an advanced/expert curve
- `curve`: Array of frequency/target/weight points defining the target response
```

#### GET `/eq/presets/optimizers`
List available optimizer configurations with different trade-offs.

**Example Request:**
```bash
curl -X GET http://localhost:10315/eq/presets/optimizers
```

**Response:**
```json
{
  "success": true,
  "count": 6,
  "optimizer_presets": [
    {
      "key": "default",
      "preset": "default",
      "name": "Default",
      "description": "Balanced optimization with moderate Q values - recommended for most applications",
      "qmax": 10.0,
      "mindb": -10.0,
      "maxdb": 3.0,
      "add_highpass": true
    },
    {
      "key": "smooth",
      "preset": "smooth",
      "name": "Smooth",
      "description": "Moderate corrections with lower Q values - forgiving of measurement errors",
      "qmax": 5.0,
      "mindb": -8.0,
      "maxdb": 2.0,
      "add_highpass": true
    },
    {
      "key": "verysmooth", 
      "preset": "verysmooth",
      "name": "Very Smooth",
      "description": "Gentle corrections with very low Q values - minimal risk of artifacts",
      "qmax": 2.0,
      "mindb": -8.0,
      "maxdb": 2.0,
      "add_highpass": true
    },
    {
      "key": "aggressive",
      "preset": "aggressive",
      "name": "Aggressive",
      "description": "Strong corrections with high Q values - requires accurate measurements",
      "qmax": 15.0,
      "mindb": -15.0,
      "maxdb": 5.0,
      "add_highpass": true
    },
    {
      "key": "precise",
      "preset": "precise",
      "name": "Precise",
      "description": "Maximum precision with highest Q values - for expert use with excellent measurements",
      "qmax": 20.0,
      "mindb": -20.0,
      "maxdb": 6.0,
      "add_highpass": true
    },
    {
      "key": "no_highpass",
      "preset": "no_highpass",
      "name": "No High-Pass",
      "description": "Default settings without automatic high-pass filter",
      "qmax": 10.0,
      "mindb": -10.0,
      "maxdb": 3.0,
      "add_highpass": false
    }
  ]
}
```

**Field Descriptions:**
- `key`: Unique identifier for the optimizer preset
- `preset`: Preset identifier (same as key)
- `name`: Human-readable name
- `description`: Detailed description of the optimization behavior
- `qmax`: Maximum Q factor allowed for filters
- `mindb`: Minimum gain reduction allowed (negative value)
- `maxdb`: Maximum gain boost allowed (positive value)
- `add_highpass`: Whether to automatically add a high-pass filter
```

### EQ Optimization Process

#### POST `/eq/optimize`
Run high-performance EQ optimization using Rust backend with real-time streaming results.

**Content-Type:** `application/json`  
**Response-Type:** `text/plain` (Server-Sent Events stream)

**Request Body Options:**

**Option 1: From Recording (Recommended)**
```json
{
  "recording_id": "abc12345",
  "target_curve": "weighted_flat",
  "optimizer_preset": "default",
  "filter_count": 8,
  "window": "hann",
  "points_per_octave": 12,
  "normalize": 1000.0
}
```

**Option 2: From FFT Data**
```json
{
  "frequencies": [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200],
  "magnitudes": [-5.2, -3.1, -2.8, -1.5, -0.8, 2.1, 3.5, 1.2, -1.8, -3.2, -4.1],
  "target_curve": "harman",
  "optimizer_preset": "smooth",
  "filter_count": 6,
  "sample_rate": 48000,
  "add_highpass": true
}
```

**Parameters:**
- `recording_id` (string, option 1): ID of completed recording to analyze
- `frequencies` (array, option 2): Frequency values in Hz
- `magnitudes` (array, option 2): Magnitude values in dB (same length as frequencies)
- `target_curve` (string): Target response curve name (see `/eq/presets/targets`)
- `optimizer_preset` (string): Optimization style (see `/eq/presets/optimizers`)
- `filter_count` (integer, optional): Number of EQ filters to generate (1-20, default: 8)
- `sample_rate` (float, optional): Audio sample rate for FFT data (default: 48000)
- `add_highpass` (boolean, optional): Override preset high-pass filter setting
- `window` (string, optional): FFT window function for recording analysis (default: "hann")
- `normalize` (float, optional): Normalization frequency for recording analysis
- `points_per_octave` (integer, optional): Frequency resolution for recording analysis (1-100, default: 12)

**Example Requests:**
```bash
# Optimize from recording with streaming results
curl -X POST "http://localhost:10315/eq/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "recording_id": "abc12345",
    "target_curve": "weighted_flat",
    "optimizer_preset": "default",
    "filter_count": 8
  }'

# Optimize from FFT data with custom parameters
curl -X POST "http://localhost:10315/eq/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "frequencies": [63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000],
    "magnitudes": [2.1, 3.8, 4.2, 3.1, 1.9, 0.8, -0.5, -1.2, -0.8, 0.2, 1.5, 2.8, 3.2, 2.1, 0.9, -1.2, -2.8, -3.5, -2.9, -1.8, -0.5, 1.2, 2.5],
    "target_curve": "harman",
    "optimizer_preset": "smooth",
    "filter_count": 6,
    "sample_rate": 48000,
    "add_highpass": true
  }'
```

**Response (Real-time Streaming Events):**
The endpoint returns a stream of Server-Sent Events with real-time optimization progress and frequency response:

```
data: {"type": "started", "message": "Starting optimization with 250 frequency points", "parameters": {"target_curve": "weighted_flat", "optimizer_preset": "default", "filter_count": 8, "sample_rate": 48000, "add_highpass": true}}

data: {"type": "initialization", "optimization_id": "abc12345", "step": 0, "message": "Detected usable frequency range: 20.0 - 20000.0 Hz (250 candidates)", "usable_range": {"f_low": 20.0, "f_high": 20000.0, "candidates": 250}, "progress": 0.0, "timestamp": 1692123456.789}

data: {"type": "filter_added", "optimization_id": "abc12345", "step": 1, "message": "Step 1: Added filter 1 at 120.0Hz, Q=1.5, 4.2dB (Error: 3.8 dB)", "filter": {"filter_type": "peaking_eq", "frequency": 120.0, "q": 1.5, "gain_db": 4.2, "description": "Peaking Eq 120Hz +4.2dB"}, "total_filters": 1, "current_filter_set": [{"filter_type": "peaking_eq", "frequency": 120.0, "q": 1.5, "gain_db": 4.2, "description": "Peaking Eq 120Hz +4.2dB", "text_format": "eq:120.0:1.500:4.20"}], "progress": 12.5, "timestamp": 1692123457.234}

data: {"type": "frequency_response", "optimization_id": "abc12345", "step": 1, "message": "Frequency response calculated for step_1", "frequency_response": {"frequencies": [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000], "magnitude_db": [0.1, 0.2, 0.4, 0.8, 1.2, 1.8, 2.5, 3.2, 4.1, 3.8, 3.2, 2.1, 1.2, 0.5, -0.2, -0.8, -1.5, -2.1, -2.8, -3.2, -3.8, -2.9, -1.8, -0.5, 0.8, 1.5, 2.2, 1.8, 1.2, 0.5], "phase_degrees": [...]}, "timestamp": 1692123457.456}

data: {"type": "filter_added", "optimization_id": "abc12345", "step": 2, "message": "Step 2: Added filter 2 at 2500.0Hz, Q=2.8, -3.8dB (Error: 2.1 dB)", "filter": {"filter_type": "peaking_eq", "frequency": 2500.0, "q": 2.8, "gain_db": -3.8, "description": "Peaking Eq 2500Hz -3.8dB"}, "total_filters": 2, "current_filter_set": [{"filter_type": "peaking_eq", "frequency": 120.0, "q": 1.5, "gain_db": 4.2, "description": "Peaking Eq 120Hz +4.2dB", "text_format": "eq:120.0:1.500:4.20"}, {"filter_type": "peaking_eq", "frequency": 2500.0, "q": 2.8, "gain_db": -3.8, "description": "Peaking Eq 2500Hz -3.8dB", "text_format": "eq:2500.0:2.800:-3.80"}], "progress": 25.0, "timestamp": 1692123458.567}

data: {"type": "frequency_response", "optimization_id": "abc12345", "step": 2, "message": "Frequency response calculated for step_2", "frequency_response": {"frequencies": [...], "magnitude_db": [...updated response...], "phase_degrees": [...]}, "timestamp": 1692123458.789}

data: {"type": "completed", "optimization_id": "abc12345", "step": 8, "message": "Optimization completed successfully", "result": {"success": true, "filters": [...], "filter_count": 8, "original_error": 8.5, "final_error": 1.9, "improvement_db": 6.6, "processing_time": 18.7, "final_frequency_response": {"frequencies": [...], "magnitude_db": [...], "phase_degrees": [...]}}, "progress": 100.0, "processing_time": 18.7, "timestamp": 1692123465.890}
```

**Frequency Response Details:**
The `frequency_response` events contain the calculated response of the current filter set:

```json
{
  "type": "frequency_response",
  "optimization_id": "abc12345", 
  "step": 1,
  "current_filter_set": [
    {
      "filter_type": "peaking_eq",
      "frequency": 120.0,
      "q": 1.5,
      "gain_db": 4.2,
      "description": "Peaking Eq 120Hz +4.2dB",
      "text_format": "eq:120.0:1.500:4.20"
    }
  ],
  "total_filters": 1,
  "frequency_response": {
    "frequencies": [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000],
    "magnitude_db": [0.1, 0.2, 0.4, 0.8, 1.2, 1.8, 2.5, 3.2, 4.1, 3.8, 3.2, 2.1, 1.2, 0.5, -0.2, -0.8, -1.5, -2.1, -2.8, -3.2, -3.8, -2.9, -1.8, -0.5, 0.8, 1.5, 2.2, 1.8, 1.2, 0.5],
    "phase_degrees": [0.0, 5.2, 12.8, 18.4, 25.1, 32.7, 38.9, 45.2, 51.8, 58.3, 64.1, 69.7, 74.8, 79.2, 82.9, 85.1, 86.8, 87.2, 86.9, 85.8, 83.7, 80.9, 77.4, 73.2, 68.5, 63.1, 57.2, 50.8, 44.1, 37.0]
  },
  "timestamp": 1692123457.456
}
```

**Important Note:** The frequency response always uses the exact same frequencies as provided in the input measurement data. If your input has 10 frequency points, the frequency response will have exactly 10 matching points. If your input has 31 points (like the curl example), the frequency response will have exactly 31 matching points. This ensures perfect alignment between the original measurement and the corrected response for accurate comparison.

The `frequency_response` events also include the `current_filter_set` and `total_filters` fields, showing which filters were applied to generate this response. This provides complete context about the optimization state at each step.

**Optimization Result Structure:**
```json
{
  "success": true,
  "filters": [
    {
      "filter_type": "peaking_eq",
      "frequency": 120.0,
      "q": 1.5,
      "gain_db": 4.2,
      "description": "Peaking Eq 120Hz +4.2dB",
      "text_format": "eq:120.0:1.500:4.20"
    },
    {
      "filter_type": "high_pass",
      "frequency": 60.0,
      "q": 0.5,
      "gain_db": 0.0,
      "description": "High Pass 60Hz",
      "text_format": "hp:60.0:0.500:0.00"
    }
  ],
  "filter_count": 8,
  "usable_range": [20.0, 20000.0, 250],
  "original_error": 8.5,
  "final_error": 1.9,
  "improvement_db": 6.6,
  "processing_time": 18.7
}
```

**Event Types:**
- `started`: Optimization initialization
- `initialization`: Usable frequency range detection
- `filter_added`: New filter added to the solution with complete list of all filters accumulated so far
- `frequency_response`: Frequency response calculated by Rust optimizer after each filter step using the same frequencies as the input measurement data
- `completed`: Optimization finished successfully with final frequency response
- `error`: Optimization failed with error message

**Optimization Features:**
- **High-Performance Rust Backend**: Fast optimization with intelligent frequency management
- **Real-time Streaming**: Live progress updates without buffering
- **Frequency Response Calculation**: Complete filter set and frequency response after each step, using the exact same frequencies as the input measurement data for accurate comparison
- **Input Frequency Preservation**: Frequency response calculations match the input measurement frequencies exactly (no interpolation or additional frequency points)
- **Frequency Deduplication**: Removes redundant frequency points for efficiency
- **Adaptive High-pass**: Intelligent high-pass filter placement
- **Usable Range Detection**: Automatically detects useful frequency range
- **Text Format Output**: Ready-to-use filter strings for EQ software

### Legacy Async API Endpoints (Deprecated)

The following endpoints are deprecated in favor of the streaming API above:

#### GET `/eq/optimize/status/<optimization_id>`
Get real-time optimization progress with detailed step information.

**Example Request:**
```bash
curl -X GET http://localhost:10315/eq/optimize/status/opt_20250818_143025_xyz789
```

**Response (In Progress):**
```json
{
  "optimization_id": "opt_20250818_143025_xyz789",
  "status": "optimizing", 
  "progress": 65.0,
  "current_step": "Optimizing filter 5/8",
  "steps_completed": 13,
  "total_steps": 20,
  "elapsed_time": 9.2,
  "estimated_remaining": 5.1,
  "current_filter": {
    "index": 5,
    "frequency": 2500.0,
    "gain_db": -3.8,
    "q": 2.1,
    "filter_type": "peaking_eq"
  },
  "intermediate_rms_error": 3.7,
  "target_rms_error": 2.0
}
```

**Response (Completed):**
```json
{
  "optimization_id": "opt_20250818_143025_xyz789", 
  "status": "completed",
  "progress": 100.0,
  "current_step": "Optimization completed",
  "steps_completed": 20,
  "total_steps": 20,
  "elapsed_time": 14.8,
  "final_rms_error": 1.9,
  "improvement_db": 8.3,
  "message": "EQ optimization completed successfully with 8 filters"
}
```

#### POST `/eq/optimize/cancel/<optimization_id>`
Cancel a running optimization process.

**Example Request:**
```bash
curl -X POST http://localhost:10315/eq/optimize/cancel/opt_20250818_143025_xyz789
```

**Response:**
```json
{
  "status": "cancelled",
  "optimization_id": "opt_20250818_143025_xyz789",
  "message": "Optimization cancelled successfully"
}
```

#### GET `/eq/optimize/result/<optimization_id>`
Get complete optimization results including generated EQ filters.

**Example Request:**
```bash
curl -X GET http://localhost:10315/eq/optimize/result/opt_20250818_143025_xyz789
```

**Response:**
```json
{
  "optimization_id": "opt_20250818_143025_xyz789",
  "status": "completed",
  "success": true,
  "target_curve": "weighted_flat",
  "optimizer_preset": "default",
  "processing_time": 14.8,
  "final_rms_error": 1.9,
  "improvement_db": 8.3,
  "filters": [
    {
      "index": 1,
      "filter_type": "peaking_eq",
      "frequency": 120.0,
      "q": 1.5,
      "gain_db": 4.2,
      "description": "Peaking EQ 120Hz 4.2dB",
      "text_format": "eq:120:1.5:4.2",
      "coefficients": {
        "b": [1.051, -1.894, 0.851],
        "a": [1.000, -1.894, 0.902]
      }
    },
    {
      "index": 2, 
      "filter_type": "peaking_eq",
      "frequency": 315.0,
      "q": 2.1,
      "gain_db": -2.8,
      "description": "Peaking EQ 315Hz -2.8dB",
      "text_format": "eq:315:2.1:-2.8",
      "coefficients": {
        "b": [0.945, -1.823, 0.883],
        "a": [1.000, -1.823, 0.828]
      }
    },
    {
      "index": 3,
      "filter_type": "peaking_eq", 
      "frequency": 2500.0,
      "q": 2.8,
      "gain_db": -3.8,
      "description": "Peaking EQ 2500Hz -3.8dB",
      "text_format": "eq:2500:2.8:-3.8",
      "coefficients": {
        "b": [0.932, -1.687, 0.756],
        "a": [1.000, -1.687, 0.688]
      }
    }
  ],
  "frequency_response": {
    "frequencies": [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400],
    "original_response": [-8.2, -7.1, -5.9, -4.8, -3.5, -2.1, -0.8, 1.2, 2.8, 3.5, 2.9, 1.8, 0.5, -1.2],
    "corrected_response": [-0.8, -0.5, -0.2, 0.1, 0.3, 0.2, -0.1, -0.3, -0.1, 0.2, 0.4, 0.1, -0.2, -0.4],
    "target_response": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  },
  "intermediate_results": [
    {
      "step": 2,
      "filters": [
        {
          "filter_type": "high_pass",
          "frequency": 60.0,
          "q": 0.5,
          "gain_db": 0.0
        },
        {
          "filter_type": "peaking_eq",
          "frequency": 120.0,
          "q": 1.5,
          "gain_db": 4.2
        }
      ],
      "improvement_db": 3.2,
      "rms_error": 5.1
    },
    {
      "step": 4,
      "filters": [
        {
          "filter_type": "high_pass",
          "frequency": 60.0,
          "q": 0.5,
          "gain_db": 0.0
        },
        {
          "filter_type": "peaking_eq",
          "frequency": 120.0,
          "q": 1.5,
          "gain_db": 4.2
        },
        {
          "filter_type": "peaking_eq",
          "frequency": 315.0,
          "q": 2.1,
          "gain_db": -2.8
        },
        {
          "filter_type": "peaking_eq",
          "frequency": 2500.0,
          "q": 2.8,
          "gain_db": -3.8
        }
      ],
      "improvement_db": 6.8,
      "rms_error": 3.5
    }
  ],
  "timestamp": "2025-08-18T14:30:40.123456"
}
```

**Note:** The `intermediate_results` array is only included when `intermediate_results_interval > 0` was specified during optimization start. Each intermediate result shows the optimization state after every n filters, allowing for progress monitoring and early termination if desired performance is reached.
```

### EQ Optimization Usage Examples

#### Complete Room Correction Workflow
```bash
#!/bin/bash
# Complete room correction measurement and optimization

API_URL="http://localhost:10315"

echo "=== Starting Room Correction Workflow ==="

# 1. Start recording for room measurement
echo "1. Starting room measurement recording..."
RECORD_RESPONSE=$(curl -s -X POST "$API_URL/audio/record/start?duration=45&sample_rate=48000")
RECORDING_ID=$(echo "$RECORD_RESPONSE" | jq -r '.recording_id')
echo "   Recording ID: $RECORDING_ID"

# 2. Generate sine sweeps for room analysis
sleep 2
echo "2. Generating sine sweeps..."
curl -s -X POST "$API_URL/audio/sweep/start?start_freq=20&end_freq=20000&duration=12&sweeps=3&amplitude=0.4" > /dev/null
echo "   3 sweeps × 12s = 36 seconds total"

# 3. Wait for recording to complete
echo "3. Waiting for recording to complete..."
while true; do
    RECORD_STATUS=$(curl -s "$API_URL/audio/record/status/$RECORDING_ID")
    COMPLETED=$(echo "$RECORD_STATUS" | jq -r '.completed')
    if [ "$COMPLETED" = "true" ]; then
        echo "   Recording completed!"
        break
    fi
    sleep 2
done

# 4. Start EQ optimization 
echo "4. Starting EQ optimization..."
OPT_RESPONSE=$(curl -s -X POST "$API_URL/eq/optimize" \
  -H "Content-Type: application/json" \
  -d "{
    \"recording_id\": \"$RECORDING_ID\",
    \"target_curve\": \"weighted_flat\", 
    \"optimizer_preset\": \"default\",
    \"filter_count\": 8,
    \"points_per_octave\": 12
  }")

OPT_ID=$(echo "$OPT_RESPONSE" | jq -r '.optimization_id')
echo "   Optimization ID: $OPT_ID"

# 5. Monitor optimization progress
echo "5. Monitoring optimization progress..."
while true; do
    OPT_STATUS=$(curl -s "$API_URL/eq/optimize/status/$OPT_ID")
    STATUS=$(echo "$OPT_STATUS" | jq -r '.status')
    PROGRESS=$(echo "$OPT_STATUS" | jq -r '.progress')
    CURRENT_STEP=$(echo "$OPT_STATUS" | jq -r '.current_step')
    
    echo "   Progress: ${PROGRESS}% - $CURRENT_STEP"
    
    if [ "$STATUS" = "completed" ]; then
        echo "   Optimization completed!"
        break
    elif [ "$STATUS" = "error" ]; then
        echo "   Optimization failed!"
        exit 1
    fi
    sleep 2
done

# 6. Get final results
echo "6. Retrieving optimization results..."
RESULT=$(curl -s "$API_URL/eq/optimize/result/$OPT_ID")

# Extract key metrics
FILTER_COUNT=$(echo "$RESULT" | jq '.filters | length')
IMPROVEMENT=$(echo "$RESULT" | jq -r '.improvement_db')
RMS_ERROR=$(echo "$RESULT" | jq -r '.final_rms_error')

echo "=== Optimization Results ==="
echo "Filters generated: $FILTER_COUNT"
echo "Improvement: ${IMPROVEMENT} dB"
echo "Final RMS error: ${RMS_ERROR} dB"

# 7. Save results and show filters
echo "$RESULT" > "eq_optimization_result_${OPT_ID}.json"
echo "Full results saved to: eq_optimization_result_${OPT_ID}.json"

echo ""
echo "=== Generated EQ Filters ==="
echo "$RESULT" | jq -r '.filters[] | "Filter \(.index): \(.filter_type) \(.frequency)Hz Q=\(.q) \(.gain_db)dB"'

echo ""
echo "=== Text Format (for import into EQ software) ==="
echo "$RESULT" | jq -r '.filters[].text_format'

echo ""
echo "Room correction workflow completed successfully!"
```

#### Quick Optimization from Existing FFT Data
```bash
# Optimize using pre-analyzed frequency response data
curl -X POST http://localhost:10315/eq/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "frequencies": [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
    "magnitudes": [3.2, 4.1, 2.8, 1.2, 0.0, -2.1, -3.8, -2.5, -1.2],
    "target_curve": "harman",
    "optimizer_preset": "smooth",
    "filter_count": 5,
    "sample_rate": 48000
  }' | jq '.optimization_id'
```

#### Monitor Multiple Optimizations
```bash
#!/bin/bash
# Monitor multiple optimization jobs

API_URL="http://localhost:10315"
OPT_IDS=("opt_id_1" "opt_id_2" "opt_id_3")

while true; do
    ALL_COMPLETE=true
    
    for OPT_ID in "${OPT_IDS[@]}"; do
        STATUS_RESPONSE=$(curl -s "$API_URL/eq/optimize/status/$OPT_ID")
        STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status')
        PROGRESS=$(echo "$STATUS_RESPONSE" | jq -r '.progress')
        
        if [ "$STATUS" != "completed" ] && [ "$STATUS" != "error" ]; then
            ALL_COMPLETE=false
            echo "$OPT_ID: ${PROGRESS}% ($STATUS)"
        else
            echo "$OPT_ID: $STATUS"
        fi
    done
    
    if [ "$ALL_COMPLETE" = true ]; then
        echo "All optimizations completed!"
        break
    fi
    
    sleep 3
done
```

## Audio Recording

### Background Recording to WAV Files

The API supports background recording to WAV files with secure file management. Recordings are stored in a temporary directory and can be managed through REST endpoints.

#### POST `/audio/record/start`
Start recording audio to a WAV file in background.

**Query Parameters:**
- `duration` (optional): Recording duration in seconds (1.0-300.0, default: 10.0)
- `device` (optional): Input device (auto-detects if not specified)
- `sample_rate` (optional): Sample rate in Hz (8000|16000|22050|44100|48000|96000, default: 48000)

**Example Request:**
```
POST /audio/record/start?duration=30&sample_rate=48000
```

**Response:**
```json
{
  "status": "started",
  "recording_id": "abc12345",
  "filename": "recording_abc12345.wav",
  "duration": 30.0,
  "device": "hw:1,0",
  "sample_rate": 48000,
  "estimated_completion": "2025-08-15T12:31:15.123456",
  "message": "Recording started: 30.0s at 48000Hz"
}
```

#### GET `/audio/record/status/<recording_id>`
Get the status of a specific recording.

**Example Request:**
```
GET /audio/record/status/abc12345
```

**Response (Active Recording):**
```json
{
  "recording_id": "abc12345",
  "status": "recording",
  "filename": "recording_abc12345.wav",
  "device": "hw:1,0",
  "duration": 30.0,
  "sample_rate": 48000,
  "elapsed_seconds": 12.5,
  "remaining_seconds": 17.5,
  "completed": false
}
```

**Response (Completed Recording):**
```json
{
  "recording_id": "abc12345",
  "status": "completed",
  "filename": "recording_abc12345.wav",
  "device": "hw:1,0",
  "duration": 30.0,
  "sample_rate": 48000,
  "timestamp": "2025-08-15T12:30:45.123456",
  "completed": true,
  "file_available": true
}
```

#### GET `/audio/record/list`
List all recordings (active and completed).

**Response:**
```json
{
  "active_recordings": [
    {
      "recording_id": "def67890",
      "status": "recording",
      "filename": "recording_def67890.wav",
      "elapsed_seconds": 5.2,
      "remaining_seconds": 24.8
    }
  ],
  "completed_recordings": [
    {
      "recording_id": "abc12345",
      "filename": "recording_abc12345.wav",
      "duration": 30.0,
      "timestamp": "2025-08-15T12:30:45.123456",
      "file_available": true
    }
  ],
  "temp_directory": "/tmp/roomeq_recordings_xyz123"
}
```

#### GET `/audio/record/download/<recording_id>`
Download a completed recording file.

**Example Request:**
```bash
curl -X GET http://localhost:10315/audio/record/download/abc12345 -o my_recording.wav
```

**Response:** Binary WAV file download

#### DELETE `/audio/record/delete/<recording_id>`
Delete a specific recording by ID.

**Example Request:**
```
DELETE /audio/record/delete/abc12345
```

**Response:**
```json
{
  "status": "deleted",
  "recording_id": "abc12345",
  "filename": "recording_abc12345.wav",
  "message": "Recording abc12345 deleted successfully"
}
```

#### DELETE `/audio/record/delete-file/<filename>`
Delete a recording file by filename (secure - only allows files in temp directory).

**Example Request:**
```
DELETE /audio/record/delete-file/recording_abc12345.wav
```

**Response:**
```json
{
  "status": "deleted",
  "filename": "recording_abc12345.wav",
  "message": "Recording file recording_abc12345.wav deleted successfully"
}
```

## Usage Examples

### Basic SPL Measurement
```bash
# Measure SPL for 1 second using auto-detected microphone
curl -X GET http://localhost:10315/spl/measure

# Measure SPL for 3 seconds using specific device
curl -X GET "http://localhost:10315/spl/measure?device=hw:1,0&duration=3.0"

# Extract just the SPL value using jq
SPL=$(curl -s http://localhost:10315/spl/measure | jq -r '.spl_db')
echo "Current SPL: ${SPL} dB"
```

### Noise Generation with Keep-Alive
```bash
# Start noise playback for 3 seconds
curl -X POST "http://localhost:10315/audio/noise/start?duration=3&amplitude=0.5"

# Extend playback by another 3 seconds (call before current duration expires)
curl -X POST "http://localhost:10315/audio/noise/keep-playing?duration=3"

# Check status
curl -X GET http://localhost:10315/audio/noise/status

# Stop playback manually
curl -X POST http://localhost:10315/audio/noise/stop
```

### Sine Sweep Generation
```bash
# Single full-spectrum sweep for room analysis
curl -X POST "http://localhost:10315/audio/sweep/start?start_freq=20&end_freq=20000&duration=10&amplitude=0.4"

# Multiple sweeps for acoustic averaging
curl -X POST "http://localhost:10315/audio/sweep/start?start_freq=100&end_freq=8000&duration=5&sweeps=3&amplitude=0.3"

# Focused frequency range for speaker testing
curl -X POST "http://localhost:10315/audio/sweep/start?start_freq=200&end_freq=2000&duration=3&amplitude=0.5"

# SoX-based sweep (uses SoX to synthesize, then plays the generated WAV)
curl -X POST "http://localhost:10315/audio/sweep/start?start_freq=20&end_freq=20000&duration=8&sweeps=2&amplitude=0.3&generator=sine_sox"

# Check sweep status (shows sweep details)
curl -X GET http://localhost:10315/audio/noise/status
```

### Audio Recording Workflow
```bash
# Start a 30-second recording at 48kHz
curl -X POST "http://localhost:10315/audio/record/start?duration=30&sample_rate=48000"
# Response includes recording_id: "abc12345"

# Check recording status
curl "http://localhost:10315/audio/record/status/abc12345"

# List all recordings
curl "http://localhost:10315/audio/record/list"

# Download completed recording
curl -X GET "http://localhost:10315/audio/record/download/abc12345" -o my_recording.wav

# Delete recording when done
curl -X DELETE "http://localhost:10315/audio/record/delete/abc12345"
```

### FFT Analysis Examples

```bash
# Basic FFT analysis of recorded file
curl -X POST "http://localhost:10315/audio/analyze/fft?filename=recording_abc12345.wav"

# FFT analysis with specific window function and normalization
curl -X POST "http://localhost:10315/audio/analyze/fft?filename=recording_abc12345.wav&window=hamming&normalize=1000"

# FFT analysis with logarithmic frequency summarization (16 points per octave)
curl -X POST "http://localhost:10315/audio/analyze/fft?filename=recording_abc12345.wav&points_per_octave=16"

# High-resolution log frequency analysis (24 points per octave) with normalization
curl -X POST "http://localhost:10315/audio/analyze/fft?filename=recording_abc12345.wav&points_per_octave=24&normalize=1000&window=hann"

# With psychoacoustic smoothing for cleaner visual presentation
curl -X POST "http://localhost:10315/audio/analyze/fft?filename=recording_abc12345.wav&psychoacoustic_smoothing=1.0&points_per_octave=16"

# Analyze specific time segment with custom FFT size
curl -X POST "http://localhost:10315/audio/analyze/fft?filename=recording_abc12345.wav&start_time=5&duration=10&fft_size=16384&points_per_octave=12"

# Alternative using start_at parameter
curl -X POST "http://localhost:10315/audio/analyze/fft?filename=recording_abc12345.wav&start_at=5&duration=10&fft_size=16384&points_per_octave=12"

# Analyze external WAV file with log frequency summarization
curl -X POST "http://localhost:10315/audio/analyze/fft?filepath=/path/to/measurement.wav&points_per_octave=16"

# Direct analysis of recording by ID (with log frequency summary and psychoacoustic smoothing)
curl -X POST "http://localhost:10315/audio/analyze/fft-recording/abc12345?points_per_octave=16&normalize=1000&psychoacoustic_smoothing=1.5"
```

**Common points_per_octave values:**
- `12`: Similar to 1/3 octave bands (121 points, 20Hz-20kHz)
- `16`: Good balance of resolution and data size (161 points)
- `24`: High resolution for detailed analysis (241 points)
- `48`: Very high resolution for research (481 points)
```

### Automated Measurement Sequence
```bash
#!/bin/bash
# Example measurement automation script

API_URL="http://localhost:10315"

echo "Starting automated acoustic measurement..."

# 1. Start background recording for full measurement
RECORD_RESPONSE=$(curl -s -X POST "$API_URL/audio/record/start?duration=60&sample_rate=48000")
RECORDING_ID=$(echo "$RECORD_RESPONSE" | grep -o '"recording_id":"[^"]*' | cut -d'"' -f4)
echo "Started recording: $RECORDING_ID"

# 2. Wait a moment, then start sine sweep
sleep 2
curl -s -X POST "$API_URL/audio/sweep/start?start_freq=20&end_freq=20000&duration=15&sweeps=3&amplitude=0.4"
echo "Started 3 sine sweeps (45 seconds total)"

# 3. Wait for sweeps to complete, then measure ambient
sleep 50
SPL_RESULT=$(curl -s "$API_URL/spl/measure?duration=5")
echo "Ambient SPL measurement: $SPL_RESULT"

# 4. Check recording status and wait for completion
RECORD_STATUS=$(curl -s "$API_URL/audio/record/status/$RECORDING_ID")
echo "Recording status: $RECORD_STATUS"

# Wait for recording to complete
while [[ $(echo "$RECORD_STATUS" | grep -o '"completed":[^,]*' | cut -d':' -f2) != "true" ]]; do
    sleep 5
    RECORD_STATUS=$(curl -s "$API_URL/audio/record/status/$RECORDING_ID")
    echo "Waiting for recording to complete..."
done

# 5. Perform FFT analysis with logarithmic frequency summarization
echo "Performing FFT analysis..."
FFT_RESULT=$(curl -s -X POST "$API_URL/audio/analyze/fft-recording/$RECORDING_ID?points_per_octave=16&normalize=1000&window=hann")
echo "FFT analysis completed with log frequency summary"

# Extract key results from FFT analysis
PEAK_FREQ=$(echo "$FFT_RESULT" | grep -o '"peak_frequency":[^,]*' | cut -d':' -f2)
PEAK_MAG=$(echo "$FFT_RESULT" | grep -o '"peak_magnitude":[^,]*' | cut -d':' -f2)
echo "Peak frequency: $PEAK_FREQ Hz at $PEAK_MAG dB"

# 6. Download recording and save FFT results
echo "Measurement sequence complete. Recording ID: $RECORDING_ID"
echo "Download with: curl '$API_URL/audio/record/download/$RECORDING_ID' -o measurement.wav"
echo "FFT results saved to fft_analysis.json"
echo "$FFT_RESULT" > "fft_analysis_$RECORDING_ID.json"
```

### Web Application Integration
```javascript
// Enhanced JavaScript class for comprehensive audio control
class RoomEQController {
    constructor(apiUrl = 'http://localhost:10315') {
        this.apiUrl = apiUrl;
        this.keepAliveInterval = null;
        this.recordingIntervals = new Map(); // Track multiple recordings
    }
    
    // Noise generation with keep-alive
    async startNoise(amplitude = 0.5, duration = 3) {
        const response = await fetch(
            `${this.apiUrl}/audio/noise/start?duration=${duration}&amplitude=${amplitude}`, 
            { method: 'POST' }
        );
        
        if (response.ok) {
            // Start keep-alive mechanism
            this.keepAliveInterval = setInterval(() => {
                this.keepPlaying();
            }, 2000);
        }
        
        return response.json();
    }
    
    async keepPlaying(duration = 3) {
        try {
            const response = await fetch(`${this.apiUrl}/audio/noise/keep-playing?duration=${duration}`, {
                method: 'POST'
            });
            return response.json();
        } catch (error) {
            console.log('Keep-alive failed:', error);
        }
    }
    
    async stopNoise() {
        if (this.keepAliveInterval) {
            clearInterval(this.keepAliveInterval);
            this.keepAliveInterval = null;
        }
        
        const response = await fetch(`${this.apiUrl}/audio/noise/stop`, {
            method: 'POST'
        });
        
        return response.json();
    }
    
    // Sine sweep generation
    async startSineSweeop(options = {}) {
        const params = new URLSearchParams({
            start_freq: options.startFreq || 20,
            end_freq: options.endFreq || 20000,
            duration: options.duration || 5,
            sweeps: options.sweeps || 1,
            amplitude: options.amplitude || 0.5,
            ...(options.device && { device: options.device })
        });
        
        const response = await fetch(
            `${this.apiUrl}/audio/sweep/start?${params}`, 
            { method: 'POST' }
        );
        
        return response.json();
    }
    
    // Audio recording
    async startRecording(duration = 30, sampleRate = 48000, device = null) {
        const params = new URLSearchParams({
            duration: duration,
            sample_rate: sampleRate,
            ...(device && { device: device })
        });
        
        const response = await fetch(
            `${this.apiUrl}/audio/record/start?${params}`, 
            { method: 'POST' }
        );
        
        const result = await response.json();
        
        if (response.ok && result.recording_id) {
            // Monitor recording progress
            this.monitorRecording(result.recording_id, duration);
        }
        
        return result;
    }
    
    monitorRecording(recordingId, duration) {
        const interval = setInterval(async () => {
            try {
                const status = await this.getRecordingStatus(recordingId);
                
                if (status.completed) {
                    clearInterval(interval);
                    this.recordingIntervals.delete(recordingId);
                    console.log(`Recording ${recordingId} completed`);
                    
                    // Trigger completion callback if set
                    if (this.onRecordingComplete) {
                        this.onRecordingComplete(recordingId, status);
                    }
                } else {
                    console.log(`Recording ${recordingId}: ${status.elapsed_seconds}s / ${status.duration}s`);
                }
            } catch (error) {
                console.error('Recording monitor error:', error);
                clearInterval(interval);
                this.recordingIntervals.delete(recordingId);
            }
        }, 1000);
        
        this.recordingIntervals.set(recordingId, interval);
    }
    
    async getRecordingStatus(recordingId) {
        const response = await fetch(`${this.apiUrl}/audio/record/status/${recordingId}`);
        return response.json();
    }
    
    async listRecordings() {
        const response = await fetch(`${this.apiUrl}/audio/record/list`);
        return response.json();
    }
    
    async downloadRecording(recordingId, filename = null) {
        const response = await fetch(`${this.apiUrl}/audio/record/download/${recordingId}`);
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename || `recording_${recordingId}.wav`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }
        
        return response.ok;
    }
    
    async deleteRecording(recordingId) {
        const response = await fetch(`${this.apiUrl}/audio/record/delete/${recordingId}`, {
            method: 'DELETE'
        });
        return response.json();
    }
    
    // FFT Analysis
    async analyzeFFT(options = {}) {
        const params = new URLSearchParams();
        
        if (options.filename) {
            params.append('filename', options.filename);
        } else if (options.filepath) {
            params.append('filepath', options.filepath);
        } else {
            throw new Error('Must specify either filename or filepath');
        }
        
        if (options.window) params.append('window', options.window);
        if (options.fftSize) params.append('fft_size', options.fftSize);
        if (options.startTime) params.append('start_time', options.startTime);
        if (options.startAt) params.append('start_at', options.startAt);
        if (options.duration) params.append('duration', options.duration);
        if (options.normalize) params.append('normalize', options.normalize);
        if (options.pointsPerOctave) params.append('points_per_octave', options.pointsPerOctave);
        
        const response = await fetch(`${this.apiUrl}/audio/analyze/fft?${params}`, {
            method: 'POST'
        });
        
        return response.json();
    }
    
    async analyzeRecordingFFT(recordingId, options = {}) {
        const params = new URLSearchParams();
        
        if (options.window) params.append('window', options.window);
        if (options.fftSize) params.append('fft_size', options.fftSize);
        if (options.startTime) params.append('start_time', options.startTime);
        if (options.startAt) params.append('start_at', options.startAt);
        if (options.duration) params.append('duration', options.duration);
        if (options.normalize) params.append('normalize', options.normalize);
        if (options.pointsPerOctave) params.append('points_per_octave', options.pointsPerOctave);
        
        const response = await fetch(`${this.apiUrl}/audio/analyze/fft-recording/${recordingId}?${params}`, {
            method: 'POST'
        });
        
        return response.json();
    }
    
    // SPL measurement
    async measureSPL(duration = 1.0, device = null) {
        const params = new URLSearchParams({
            duration: duration,
            ...(device && { device: device })
        });
        
        const response = await fetch(`${this.apiUrl}/spl/measure?${params}`);
        return response.json();
    }
    
    // Status monitoring
    async getPlaybackStatus() {
        const response = await fetch(`${this.apiUrl}/audio/noise/status`);
        return response.json();
    }
    
    // Cleanup
    cleanup() {
        if (this.keepAliveInterval) {
            clearInterval(this.keepAliveInterval);
        }
        
        // Clear all recording monitors
        this.recordingIntervals.forEach(interval => clearInterval(interval));
        this.recordingIntervals.clear();
        
        // Stop any active playback
        this.stopNoise();
    }
}

// Usage examples
const roomEQ = new RoomEQController();

// Set recording completion callback with FFT analysis
roomEQ.onRecordingComplete = async (recordingId, status) => {
    console.log(`Recording ${recordingId} completed: ${status.filename}`);
    
    // Perform FFT analysis with logarithmic frequency summarization
    try {
        const fftResult = await roomEQ.analyzeRecordingFFT(recordingId, {
            pointsPerOctave: 16,
            normalize: 1000,
            window: 'hann'
        });
        
        console.log('FFT Analysis Results:', {
            peakFreq: fftResult.fft_analysis.peak_frequency,
            peakMag: fftResult.fft_analysis.peak_magnitude,
            logSummaryPoints: fftResult.fft_analysis.log_frequency_summary?.n_points
        });
        
        // Process log frequency data for visualization
        if (fftResult.fft_analysis.log_frequency_summary) {
            const logData = fftResult.fft_analysis.log_frequency_summary;
            console.log(`Log frequency analysis: ${logData.n_points} points over ${logData.n_octaves} octaves`);
        }
    } catch (error) {
        console.error('FFT analysis failed:', error);
    }
    
    // Auto-download recording
    roomEQ.downloadRecording(recordingId);
};

// Example: Room response measurement
async function measureRoomResponse() {
    try {
        // Start recording for full measurement
        const recording = await roomEQ.startRecording(60, 48000);
        console.log('Recording started:', recording.recording_id);
        
        // Wait 2 seconds, then generate test signal
        setTimeout(async () => {
            const sweep = await roomEQ.startSineSweeop({
                startFreq: 20,
                endFreq: 20000,
                duration: 10,
                sweeps: 3,
                amplitude: 0.4
            });
            console.log('Sine sweeps started:', sweep);
            
            // Measure ambient after sweeps complete
            setTimeout(async () => {
                const spl = await roomEQ.measureSPL(3.0);
                console.log('Ambient level:', spl.spl_db, 'dB SPL');
            }, 35000); // After 3 sweeps complete
            
        }, 2000);
        
    } catch (error) {
        console.error('Measurement error:', error);
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    roomEQ.cleanup();
});
```

## Error Handling

The API uses standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (no active playback to extend/stop)
- `500`: Internal Server Error

Error responses include a `detail` field with error description:

```json
{
  "detail": "Failed to start noise: ALSA device not found"
}
```

## Keep-Alive Mechanism Details

The keep-alive mechanism is designed for web applications that need continuous audio playback:

1. **Start playback** with `/audio/noise/start?duration=3`
2. **Extend playback** every 2-3 seconds with `/audio/noise/keep-playing?duration=3`
3. **Automatic stop** occurs if no keep-alive request is received before the stop time
4. **Manual stop** available with `/audio/noise/stop`

This ensures that audio stops within 3 seconds of a web browser closing or losing connectivity, preventing runaway audio playback.

## Hardware Requirements

- **Input devices**: USB microphones or built-in microphones with ALSA support
- **Output devices**: Any ALSA-compatible audio output device
- **Microphone calibration**: Sensitivity values should be configured in ALSA mixer settings

## Starting the Server

```bash
# Start server using the roomeq-server command (recommended)
roomeq-server

# Or start manually with Python
cd /path/to/roomeq
PYTHONPATH=src python3 -m roomeq.roomeq_server

# Or using the main function directly
cd /path/to/roomeq
python3 -c "import sys; sys.path.insert(0, 'src'); from roomeq.roomeq_server import main; main()"
```

The server will be available at:
- **API endpoints**: `http://localhost:10315`
- **API documentation**: `http://localhost:10315/` (comprehensive endpoint documentation)
- **Version information**: `http://localhost:10315/version`

## Technical Implementation

### Framework Details
- **Framework**: Flask with CORS support for cross-origin requests
- **Threading**: Multi-threaded request handling for concurrent operations
- **Audio Backend**: ALSA with arecord subprocess fallback for compatibility
- **Recording Storage**: Secure temporary directory with automatic cleanup
- **File Security**: Path validation and access controls for recording management

### Audio Specifications
- **Recording Format**: 16-bit PCM WAV files
- **Sample Rates**: 8000, 16000, 22050, 44100, 48000, 96000 Hz
- **Sweep Type**: Logarithmic frequency progression for acoustic analysis
- **Noise Type**: White noise with uniform frequency distribution
- **Calibration**: Automatic microphone sensitivity and gain compensation

### Security Features
- **File Access**: Recording files restricted to temporary directory
- **Path Validation**: Directory traversal protection
- **File Extensions**: Only .wav files allowed for security
- **Cleanup**: Automatic temp directory creation and management

## System Integration

This API can be integrated into:

### Acoustic Measurement Applications
- **Room Response Analysis**: Full spectrum sine sweeps with multiple averaging
- **Speaker Testing**: Focused frequency range measurements
- **Microphone Calibration**: SPL measurement with known reference signals
- **Environmental Monitoring**: Continuous background recording and SPL monitoring

### Software Integration
- **Room Correction Software**: REW (Room EQ Wizard), Acourate, DRC-FIR integration
- **Audio Testing Frameworks**: Automated test suites with measurement validation
- **Web-based Interfaces**: Browser-based measurement and control applications
- **Monitoring Systems**: Long-term acoustic monitoring with scheduled recordings
- **Research Applications**: Acoustic research with programmatic control and data collection

### Example Integration Workflows

**Room Correction Workflow:**
1. Start background recording for reference
2. Generate calibrated sine sweeps with multiple averages
3. Measure ambient noise levels
4. Download recording for analysis in room correction software
5. Repeat measurements with different positions/configurations

**Automated Testing:**
1. Schedule periodic SPL measurements
2. Generate test signals at specific times
3. Record system responses for validation
4. Download and process recordings automatically
5. Generate reports with measurement data

**Web Dashboard Integration:**
- Real-time SPL monitoring display
- Interactive sweep generation controls
- Recording management interface
- Download and playback capabilities
- Status monitoring for all operations

The RESTful design makes it easy to integrate with any programming language or framework that supports HTTP requests, including Python, JavaScript, MATLAB, LabVIEW, and command-line tools.
