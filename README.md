# RoomEQ Audio Processing System

A comprehensive REST API system for microphone detection, SPL measurement, and audio signal generation designed for acoustic testing and room correction applications.

## Quick Start

### Installation
```bash
# Install required Python packages
sudo apt install python3-alsaaudio python3-numpy python3-fastapi python3-uvicorn

# Clone and navigate to project
cd /path/to/roomeq
```

### Start the Server
```bash
# Start on default port 10315
python3 start_server.py

# Access API documentation
# http://localhost:10315/docs
```

## Core Features

- 🎤 **Automatic microphone detection** with sensitivity and gain calibration
- 📊 **Accurate SPL measurement** using calibrated microphones  
- 🔊 **Signal generation** with white noise and keep-alive control
- 🌐 **REST API** with comprehensive documentation
- ⚡ **Real-time control** for continuous testing scenarios

## Quick Usage Examples

### Measure SPL
```bash
# Auto-detect microphone and measure for 1 second
curl "http://localhost:10315/spl/measure"

# Use specific device for 3 seconds
curl "http://localhost:10315/spl/measure?device=hw:1,0&duration=3"
```

### Generate Test Signals
```bash
# Start noise playback
curl -X POST "http://localhost:10315/audio/noise/start?amplitude=0.5&duration=5"

# Keep playing (call every 2-3 seconds for continuous playback)
curl -X POST "http://localhost:10315/audio/noise/keep-playing?duration=3"

# Stop playback
curl -X POST "http://localhost:10315/audio/noise/stop"
```

### Check Available Devices
```bash
# List microphones with sensitivity info
curl "http://localhost:10315/microphones"

# List all audio cards
curl "http://localhost:10315/audio/cards"
```

## Command Line Tools

### SPL Meter
```bash
# Basic SPL measurement
python3 spl_meter.py

# Specific device with verbose output
python3 spl_meter.py -d hw:1,0 -t 2.0 -v
```

### Signal Generator
```bash
# Generate white noise for 5 seconds
python3 signal_gen.py noise -t 5 -a 0.3

# Generate sine sweep from 20Hz to 20kHz over 10 seconds  
python3 signal_gen.py sweep -s 20 -e 20000 -t 10
```

### Acoustic Testing
```bash
# Run calibration test sequence
python3 acoustic_test.py --test calibration

# Test with specific signal
python3 acoustic_test.py --test noise --duration 5 --amplitude 0.4
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and endpoint overview |
| `/docs` | GET | Interactive API documentation (Swagger UI) |
| `/microphones` | GET | Detected microphones with calibration data |
| `/spl/measure` | GET | Measure sound pressure level |
| `/audio/noise/start` | POST | Start noise playback with timeout |
| `/audio/noise/keep-playing` | POST | Extend current playback (keep-alive) |
| `/audio/noise/stop` | POST | Stop current playback |
| `/audio/noise/status` | GET | Get playback status |

## Web Application Integration

The keep-alive mechanism is perfect for web applications:

```javascript
// Start noise with automatic keep-alive
class AudioController {
    async startNoise() {
        await fetch('/audio/noise/start?duration=3', {method: 'POST'});
        this.keepAlive = setInterval(() => {
            fetch('/audio/noise/keep-playing?duration=3', {method: 'POST'});
        }, 2000);
    }
    
    async stopNoise() {
        clearInterval(this.keepAlive);
        await fetch('/audio/noise/stop', {method: 'POST'});
    }
}
```

## Configuration

### Microphone Sensitivity
The system automatically detects microphone sensitivity from ALSA mixer settings. For manual configuration:

```bash
# Set microphone gain (affects measurement calibration)
amixer -c 1 cset name='Mic Capture Volume' 20dB
```

### Port Configuration
Default port is 10315. To change:

```bash
# Start on different port
PYTHONPATH=src uvicorn roomeq.roomeq_server:app --host 0.0.0.0 --port 8080
```

## Hardware Compatibility

- **Microphones**: USB microphones with ALSA support (e.g., Dayton UMM-6)
- **Audio outputs**: Any ALSA-compatible sound card
- **Operating systems**: Linux with ALSA support

## Files Structure

```
roomeq/
├── start_server.py              # Server starter script  
├── spl_meter.py                 # Command-line SPL measurement
├── signal_gen.py                # Command-line signal generator
├── acoustic_test.py             # Acoustic testing suite
├── doc/
│   └── api.md                   # Complete API documentation
└── src/roomeq/
    ├── analysis.py              # Audio analysis and SPL measurement
    ├── microphone.py            # Microphone detection and calibration
    ├── signal_generator.py      # Signal generation functionality
    └── roomeq_server.py         # FastAPI REST server
```

## Use Cases

- **Room correction**: Measure room response and generate test signals
- **Audio testing**: Automated testing of audio equipment
- **Acoustic measurement**: Professional SPL measurement with calibrated microphones
- **Web applications**: Browser-based measurement interfaces
- **Quality control**: Automated audio testing in production

## Documentation

- **Complete API docs**: See [doc/api.md](doc/api.md)
- **Interactive docs**: http://localhost:10315/docs (when server is running)
- **Alternative docs**: http://localhost:10315/redoc

## Troubleshooting

### No Microphones Detected
```bash
# Check ALSA devices
arecord -l

# Test microphone manually  
arecord -D hw:1,0 -d 2 -f S16_LE -r 48000 test.wav
```

### Permission Issues
```bash
# Add user to audio group
sudo usermod -a -G audio $USER
# Logout and login again
```

### Server Won't Start
```bash
# Check if port is available
netstat -tuln | grep 10315

# Start with different port
PYTHONPATH=src uvicorn roomeq.roomeq_server:app --host 0.0.0.0 --port 8080
```

## License

See LICENSE file for details.
