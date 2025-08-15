# RoomEQ Audio Processing API Documentation

## Overview

The RoomEQ Audio Processing API provides a comprehensive REST interface for microphone detection, sound pressure level (SPL) measurement, audio signal generation, recording, and FFT analysis functionality. This API is designed for acoustic measurement systems, room correction, and audio testing applications.

**Base URL:** `http://localhost:10315`  
**API Version:** 0.3.0  
**Framework:** Flask with CORS support
**Documentation:** Available at the root endpoint `/`

## Features

- **Microphone Detection**: Automatic detection of USB and built-in microphones with sensitivity and gain information
- **SPL Measurement**: Accurate sound pressure level measurement with calibrated microphones
- **Signal Generation**: White noise and logarithmic sine sweep generation with keep-alive functionality
- **Multiple Sweep Support**: Generate consecutive sine sweeps for acoustic averaging
- **Audio Recording**: Background recording to WAV files with secure file management
- **FFT Analysis**: Comprehensive spectral analysis of WAV files with windowing functions, normalization, logarithmic frequency summarization, peak detection, and frequency band analysis
- **Real-time Control**: Start, stop, extend, and monitor audio operations through REST endpoints
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

**Response:**
```json
{
  "version": "0.3.0",
  "api_name": "RoomEQ Audio Processing API",
  "features": [
    "Microphone detection with sensitivity and gain",
    "SPL measurement with microphone calibration",
    "White noise generation with keep-alive control",
    "Logarithmic sine sweep generation with multiple repeat support",
    "Background audio recording to WAV files",
    "Real-time playback and recording management",
    "Cross-Origin Resource Sharing (CORS) support"
  ]
}
```

### Microphone Detection

#### GET `/microphones`
Get detected microphones with detailed properties.

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

**Example Request:**
```
GET /spl/measure?device=hw:1,0&duration=2.0
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

**Example Request:**
```
POST /audio/noise/start?duration=5&amplitude=0.3&device=hw:0,0
```

**Response:**
```json
{
  "status": "started",
  "duration": 5.0,
  "amplitude": 0.3,
  "device": "hw:0,0",
  "stop_time": "2025-08-14T15:30:45.123456",
  "message": "Noise playback started for 5.0 seconds"
}
```

#### POST `/audio/noise/keep-playing`
Extend current noise playback duration (keep-alive mechanism).

**Query Parameters:**
- `duration` (optional): Additional duration in seconds (1.0-30.0, default: 3.0)

**Example Request:**
```
POST /audio/noise/keep-playing?duration=3
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

### Sine Sweep Generation

#### POST `/audio/sweep/start`
Start logarithmic sine sweep(s) with optional multiple repeat support.

**Query Parameters:**
- `start_freq` (optional): Starting frequency in Hz (10-22000, default: 20)
- `end_freq` (optional): Ending frequency in Hz (10-22000, default: 20000)
- `duration` (optional): Duration per sweep in seconds (1.0-30.0, default: 5.0)
- `sweeps` (optional): Number of consecutive sweeps (1-10, default: 1)
- `amplitude` (optional): Amplitude level (0.0-1.0, default: 0.5)
- `device` (optional): Output device (e.g., "hw:0,0"). Uses default if not specified

**Example Requests:**
```bash
# Single sweep - full spectrum, 10 seconds
POST /audio/sweep/start?start_freq=20&end_freq=20000&duration=10&amplitude=0.4

# Multiple sweeps for averaging - 3 consecutive sweeps
POST /audio/sweep/start?start_freq=100&end_freq=8000&duration=5&sweeps=3&amplitude=0.3
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
  "device": "default",
  "stop_time": "2025-08-15T12:30:45.123456",
  "message": "3 sine sweep(s) started: 100.0 Hz → 8000.0 Hz, 5.0s each (total: 15.0s)"
}
```

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
- `duration` (float, optional): Duration to analyze in seconds (default: entire file from start_time)
- `normalize` (float, optional): Frequency in Hz to normalize to 0 dB (all other levels adjusted relative to this frequency)
- `points_per_octave` (integer, optional): Summarize FFT into logarithmic frequency buckets (1-100, enables log frequency summarization)

**Note:** Must specify either `filename` OR `filepath`, not both.

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

### Logarithmic Frequency Summarization

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
- `duration` (float, optional): Duration to analyze in seconds
- `normalize` (float, optional): Frequency in Hz to normalize to 0 dB
- `points_per_octave` (integer, optional): Summarize FFT into logarithmic frequency buckets (1-100)

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
curl "http://localhost:10315/spl/measure"

# Measure SPL for 3 seconds using specific device
curl "http://localhost:10315/spl/measure?device=hw:1,0&duration=3.0"
```

### Noise Generation with Keep-Alive
```bash
# Start noise playback for 3 seconds
curl -X POST "http://localhost:10315/audio/noise/start?duration=3&amplitude=0.5"

# Extend playback by another 3 seconds (call before current duration expires)
curl -X POST "http://localhost:10315/audio/noise/keep-playing?duration=3"

# Check status
curl "http://localhost:10315/audio/noise/status"

# Stop playback
curl -X POST "http://localhost:10315/audio/noise/stop"
```

### Sine Sweep Generation
```bash
# Single full-spectrum sweep for room analysis
curl -X POST "http://localhost:10315/audio/sweep/start?start_freq=20&end_freq=20000&duration=10&amplitude=0.4"

# Multiple sweeps for acoustic averaging
curl -X POST "http://localhost:10315/audio/sweep/start?start_freq=100&end_freq=8000&duration=5&sweeps=3&amplitude=0.3"

# Focused frequency range for speaker testing
curl -X POST "http://localhost:10315/audio/sweep/start?start_freq=200&end_freq=2000&duration=3&amplitude=0.5"

# Check sweep status (shows sweep details)
curl "http://localhost:10315/audio/noise/status"
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

# Analyze specific time segment with custom FFT size
curl -X POST "http://localhost:10315/audio/analyze/fft?filename=recording_abc12345.wav&start_time=5&duration=10&fft_size=16384&points_per_octave=12"

# Analyze external WAV file with log frequency summarization
curl -X POST "http://localhost:10315/audio/analyze/fft?filepath=/path/to/measurement.wav&points_per_octave=16"

# Direct analysis of recording by ID (with log frequency summary)
curl -X POST "http://localhost:10315/audio/analyze/fft-recording/abc12345?points_per_octave=16&normalize=1000"
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
