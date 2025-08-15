# RoomEQ Audio Processing API Documentation

## Overview

The RoomEQ Audio Processing API provides a REST interface for microphone detection, sound pressure level (SPL) measurement, and audio signal generation. This API is designed for acoustic measurement systems, room correction, and audio testing applications.

**Base URL:** `http://localhost:10315`  
**API Version:** 0.2.0  
**Documentation:** `http://localhost:10315/docs` (Interactive Swagger UI)  
**ReDoc:** `http://localhost:10315/redoc` (Alternative documentation)

## Features

- **Microphone Detection**: Automatic detection of USB and built-in microphones with sensitivity and gain information
- **SPL Measurement**: Accurate sound pressure level measurement with calibrated microphones
- **Signal Generation**: White noise generation with keep-alive functionality for continuous testing
- **Real-time Control**: Start, stop, and extend audio playback through REST endpoints

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
  "version": "0.2.0",
  "description": "REST API for microphone detection, SPL measurement, and audio signal generation",
  "endpoints": {
    "info": ["/", "/version", "/docs", "/redoc"],
    "microphones": ["/microphones", "/microphones/raw"],
    "audio_devices": ["/audio/inputs", "/audio/cards"],
    "measurements": ["/spl/measure"],
    "signal_generation": [
      "/audio/noise/start",
      "/audio/noise/keep-playing", 
      "/audio/noise/stop",
      "/audio/noise/status"
    ]
  },
  "usage": {
    "start_noise": "POST /audio/noise/start?duration=3&amplitude=0.5",
    "keep_playing": "POST /audio/noise/keep-playing?duration=3",
    "measure_spl": "GET /spl/measure?duration=1.0"
  }
}
```

#### GET `/version`
Get detailed version information.

**Response:**
```json
{
  "version": "0.2.0",
  "api_name": "RoomEQ Audio Processing API",
  "features": [
    "Microphone detection with sensitivity and gain",
    "SPL measurement",
    "White noise generation with keep-alive control",
    "Real-time playback management"
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
- `active`: Whether noise is currently playing
- `amplitude`: Current playback amplitude
- `device`: Output device being used
- `remaining_seconds`: Seconds until automatic stop
- `stop_time`: ISO timestamp when playback will stop

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

### Web Application Integration
```javascript
// JavaScript example for web application
class AudioController {
    constructor(apiUrl = 'http://localhost:10315') {
        this.apiUrl = apiUrl;
        this.keepAliveInterval = null;
    }
    
    async startNoise(amplitude = 0.5) {
        const response = await fetch(
            `${this.apiUrl}/audio/noise/start?duration=3&amplitude=${amplitude}`, 
            { method: 'POST' }
        );
        
        if (response.ok) {
            // Start keep-alive mechanism - send request every 2 seconds
            this.keepAliveInterval = setInterval(() => {
                this.keepPlaying();
            }, 2000);
        }
        
        return response.json();
    }
    
    async keepPlaying() {
        try {
            await fetch(`${this.apiUrl}/audio/noise/keep-playing?duration=3`, {
                method: 'POST'
            });
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
    
    async measureSPL(duration = 1.0) {
        const response = await fetch(
            `${this.apiUrl}/spl/measure?duration=${duration}`
        );
        return response.json();
    }
}

// Usage
const audio = new AudioController();

// Start noise and measure SPL
audio.startNoise(0.3).then(() => {
    console.log('Noise started');
    
    // Measure after 1 second
    setTimeout(() => {
        audio.measureSPL(2.0).then(result => {
            console.log(`SPL: ${result.spl_db} dB`);
        });
    }, 1000);
});

// Stop when page unloads
window.addEventListener('beforeunload', () => {
    audio.stopNoise();
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
# Start server on default port 10315
cd /path/to/roomeq
python3 start_server.py

# Or start manually with uvicorn
PYTHONPATH=src uvicorn roomeq.roomeq_server:app --host 0.0.0.0 --port 10315
```

The server will be available at:
- API endpoints: `http://localhost:10315`
- Interactive documentation: `http://localhost:10315/docs`
- Alternative docs: `http://localhost:10315/redoc`

## System Integration

This API can be integrated into:
- Room correction software
- Acoustic measurement applications  
- Audio testing frameworks
- Web-based measurement interfaces
- Automated test suites

The RESTful design makes it easy to integrate with any programming language or framework that supports HTTP requests.
