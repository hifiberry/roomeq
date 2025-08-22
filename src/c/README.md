# HiFiBerry C Audio Analysis Tools

This directory contains C-based audio analysis tools imported from the previous HiFiBerry OS release.

## Source

These files were imported from the HiFiBerry OS repository:
- **Repository**: https://github.com/hifiberry/hifiberry-os
- **Source Path**: `buildroot/package/hifiberry-measurements/analyze/`
- **Import Date**: August 2025
- **Original Version**: HiFiBerry MultiFFT Tool V0.9 (c) 2019

## Files

### Core Analysis Tool
- **`analyze.c`** - FFT analysis implementation by Joerg Schambacher (i2Audio GmbH)
- **`fft-analyse`** - Compiled binary (renamed from original `analyze`)
- **`Makefile`** - Build configuration (updated for portability)

### Measurement Scripts
- **`record-sweep`** - Records frequency sweep measurements
- **`room-measure`** - Automated multi-measurement analysis workflow

## Purpose

These tools provide acoustic measurement capabilities for room correction and audio analysis:

1. **FFT Analysis**: Process recorded audio files to extract frequency response
2. **Sweep Recording**: Generate and record frequency sweeps for acoustic measurement
3. **Room Measurement**: Automated workflow for multiple measurements with averaging

## Usage

### FFT Analysis
```bash
fft-analyse -r reference-signal.wav recording1.wav recording2.wav
```

### Record Sweep
```bash
record-sweep hw:0,0 left
```

### Room Measurement
```bash
room-measure hw:0,0 left 8
```

## Integration

These tools are integrated into the roomeq debian package and installed system-wide at:
- `/usr/bin/fft-analyse`
- `/usr/bin/record-sweep` 
- `/usr/bin/room-measure`

Man pages are provided for all tools.

## Dependencies

- **Build**: gcc, libasound2-dev
- **Runtime**: sox, alsa-utils, libasound2

## License

Original HiFiBerry tools - see individual file headers for copyright information.
