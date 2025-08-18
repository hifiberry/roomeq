# Room EQ - Audio Analysis System

A comprehensive audio processing system for acoustic measurement, room correction, and audio equipment testing. Features FFT analysis with logarithmic frequency summarization, mathematically accurate sine sweep generation, and SPL measurement capabilities.

## What It Does

**Room EQ** provides audio analysis tools for:

- **Audio Equipment Testing** - Measure frequency response of speakers, headphones, amplifiers, and audio systems
- **Room Acoustics** - Analyze room frequency response, identify standing waves and acoustic issues  
- **Quality Assurance** - Automated testing for audio equipment production and verification
- **Research & Development** - Validate signal processing algorithms and filter designs

## Key Capabilities

 - **FFT Analysis** - Logarithmic frequency bucketing optimized for audio analysis  
 - **SPL Measurement** - Calibrated microphone integration with automatic sensitivity detection  
 - **Test Signals** - Sine sweeps with flat frequency response and calibrated white noise  
 - **REST API** - Complete web API for automation and integration  
 - **Real-time Control** - Live monitoring and continuous measurement capabilities  
 - **Validated Accuracy** - Results verified against industry standards  

## Quick Start

### Installation
```bash
# Install dependencies
sudo apt install python3-alsaaudio python3-numpy python3-scipy python3-flask python3-flask-cors

# Start the server
python3 start_server.py
```

### Basic Usage
```python
# FFT analysis with logarithmic frequency summarization
from roomeq.fft import analyze_wav_file

result = analyze_wav_file("audio.wav", points_per_octave=16)
frequencies = result['fft_analysis']['log_frequency_summary']['frequencies']
magnitudes = result['fft_analysis']['log_frequency_summary']['magnitudes']
```

### Web API
```bash
# Measure SPL
curl "http://localhost:10315/spl/measure"

# Analyze audio file
curl -X POST -F "file=@audio.wav" -F "points_per_octave=16"  
     "http://localhost:10315/api/analyze"

# Generate test signals  
curl -X POST "http://localhost:10315/audio/noise/start?duration=5"
```

## Applications

### Audio Equipment Testing
Measure frequency response, distortion, and performance characteristics of:
- Loudspeakers and headphones
- Amplifiers and audio interfaces
- Microphones and recording equipment
- Audio processing equipment

### Room Acoustics Analysis  
Analyze and optimize acoustic environments:
- Room frequency response measurement
- Standing wave and resonance identification
- Reverberation time analysis
- Acoustic treatment verification

### Quality Control & Production
Automated testing solutions for:
- Production line quality control
- Specification compliance verification
- Batch testing and statistical analysis
- Performance regression testing

## Core Features

### FFT Analysis Engine
- **Logarithmic Frequency Bucketing**: Configure points per octave for optimal audio analysis
- **Windowing Functions**: Hann, Hamming, Blackman windowing functions
- **Automatic Normalization**: Optional frequency-based normalization
- **Peak Detection**: Automatic identification of spectral features

*Mathematical details in [FFT_DOCUMENTATION.md](FFT_DOCUMENTATION.md)*

### Signal Generation
- **Sine Sweeps**: Logarithmic frequency progression with amplitude compensation
- **White Noise**: Calibrated test signals with precise amplitude control
- **Keep-alive Control**: Continuous signal generation for extended testing

### SPL Measurement
- **Automatic Microphone Detection**: Smart detection with sensitivity calibration
- **Real-time Monitoring**: Continuous SPL measurement capabilities
- **Hardware Integration**: Support for USB microphones and audio interfaces

## Command Line Tools

```bash
# SPL measurement
python3 spl_meter.py -d hw:1,0 -t 5.0

# Signal generation
python3 signal_gen.py noise -t 10 -a 0.5
python3 signal_gen.py sweep -s 20 -e 20000 -t 10

# Acoustic testing
python3 acoustic_test.py --test calibration
```

## Hardware Support

- **Microphones**: USB microphones with ALSA support (Dayton UMM-6, etc.)
- **Audio Interfaces**: Audio interfaces and sound cards
- **Operating Systems**: Linux with ALSA audio subsystem
- **File Formats**: WAV files and real-time audio streams

## Integration

Well suited for:
- **Web Applications**: Browser-based measurement interfaces
- **Automated Testing**: CI/CD integration for audio product testing  
- **Research Tools**: Academic and commercial research applications
- **Production Systems**: Quality control in manufacturing environments

```javascript
// Example web integration
const formData = new FormData();
formData.append('file', audioFile);
formData.append('points_per_octave', '16');

const response = await fetch('/api/analyze', {
    method: 'POST',
    body: formData
});
```

## Documentation

- **[FFT_DOCUMENTATION.md](FFT_DOCUMENTATION.md)** - Mathematical algorithms and technical details
- **[doc/api.md](doc/api.md)** - Complete API reference
- **Interactive API docs** - http://localhost:10315/docs (when server running)

## Quick Troubleshooting

```bash
# Check audio devices
arecord -l

# Test microphone
arecord -D hw:1,0 -d 2 -f S16_LE -r 48000 test.wav

# Verify installation
python3 -c "from roomeq.fft import analyze_wav_file; print('Ready!')"
```

---

**Audio analysis system - ready for research, development, and production use.**
