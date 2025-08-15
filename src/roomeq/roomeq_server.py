#!/usr/bin/env python3
from flask import Flask, jsonify, request, abort, send_file
from flask_cors import CORS
from typing import List, Dict, Any, Optional
import logging
import time
import os
import numpy as np
import wave
import signal
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# Local imports
from roomeq.fft import load_wav_file, compute_fft, analyze_wav_file, validate_fft_parameters
from roomeq.recording import (
    recording_manager, 
    start_recording, 
    get_recording_status,
    list_recordings,
    delete_recording,
    delete_recording_file,
    validate_recording_file,
    cleanup_old_recordings,
    start_cleanup_timer,
    stop_cleanup_timer,
    get_cleanup_status,
    signal_handler as recording_signal_handler
)

from .microphone import MicrophoneDetector, detect_microphones
from .analysis import measure_spl
from .signal_generator import SignalGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global signal generator and playback state
_signal_generator = None
_playback_state = {
    'active': False,
    'stop_time': None,
    'amplitude': 0.5,
    'device': None,
    'signal_type': 'noise',  # 'noise' or 'sine_sweep'
    'start_freq': None,
    'end_freq': None,
    'sweeps': None,
    'sweep_duration': None,
    'total_duration': None
}

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Add response headers middleware
@app.after_request
def after_request(response):
    """Add headers to all responses for better proxy compatibility."""
    response.headers["connection"] = "close"
    response.headers["server"] = "roomeq-api/0.3.0"
    return response

# Add request logging middleware
@app.before_request
def log_request_info():
    """Log detailed request information for debugging."""
    logger.info(f"Request: {request.method} {request.url}")
    if request.args:
        logger.debug(f"Query parameters: {dict(request.args)}")
    if request.json:
        logger.debug(f"JSON payload: {request.json}")

# Add error logging
@app.errorhandler(404)
def not_found_error(error):
    """Enhanced 404 error handler with detailed logging."""
    logger.warning(f"404 Not Found: {request.method} {request.url} - {error.description}")
    return jsonify({
        "error": "Not Found",
        "message": error.description or "The requested resource was not found",
        "endpoint": request.endpoint,
        "url": request.url,
        "method": request.method
    }), 404

@app.errorhandler(400)
def bad_request_error(error):
    """Enhanced 400 error handler with detailed logging."""
    logger.warning(f"400 Bad Request: {request.method} {request.url} - {error.description}")
    return jsonify({
        "error": "Bad Request", 
        "message": error.description or "The request was invalid",
        "endpoint": request.endpoint,
        "url": request.url,
        "method": request.method
    }), 400


def validate_float_param(param_name: str, value: str, min_val: float = None, max_val: float = None) -> float:
    """Validate and convert a string parameter to float with optional bounds checking."""
    try:
        val = float(value)
        if min_val is not None and val < min_val:
            abort(400, f"{param_name} must be >= {min_val}")
        if max_val is not None and val > max_val:
            abort(400, f"{param_name} must be <= {max_val}")
        return val
    except ValueError:
        abort(400, f"Invalid {param_name}: must be a number")


@app.route("/version", methods=["GET"])
def get_version():
    """Get API version information."""
    return jsonify({
        "version": "0.3.0",
        "api_name": "RoomEQ Audio Processing API",
        "features": [
            "Microphone detection with sensitivity and gain",
            "SPL measurement",
            "White noise generation with keep-alive control",
            "Multiple consecutive sine sweeps",
            "Background audio recording to WAV files",
            "FFT spectral analysis with normalization",
            "Frequency band analysis and peak detection",
            "Real-time playback and recording management"
        ]
    })


@app.route("/microphones", methods=["GET"])
def get_microphones():
    """Get detected microphones with their properties."""
    try:
        detector = MicrophoneDetector()
        microphones = detector.detect_microphones()
        
        result = []
        for card_index, device_name, sensitivity, gain_db in microphones:
            result.append({
                "card_index": card_index,
                "device_name": device_name,
                "sensitivity": float(sensitivity) if sensitivity != "0" else 0.0,
                "sensitivity_str": sensitivity,
                "gain_db": gain_db
            })
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error detecting microphones: {e}")
        abort(500, f"Failed to detect microphones: {str(e)}")


@app.route("/microphones/raw", methods=["GET"])
def get_microphones_raw():
    """Get detected microphones in raw format (compatible with bash script output)."""
    try:
        microphones = detect_microphones()
        return jsonify({"microphones": microphones})
    
    except Exception as e:
        logger.error(f"Error detecting microphones: {e}")
        abort(500, f"Failed to detect microphones: {str(e)}")


@app.route("/audio/inputs", methods=["GET"])
def get_audio_inputs():
    """Get audio input card indices."""
    try:
        detector = MicrophoneDetector()
        input_cards = detector.get_audio_inputs()
        
        return jsonify({
            "input_cards": input_cards,
            "count": len(input_cards)
        })
    
    except Exception as e:
        logger.error(f"Error getting audio inputs: {e}")
        abort(500, f"Failed to get audio inputs: {str(e)}")


@app.route("/audio/cards", methods=["GET"])
def get_audio_cards():
    """Get all available audio cards."""
    try:
        detector = MicrophoneDetector()
        cards = detector.audio_cards
        
        return jsonify({
            "cards": cards,
            "count": len(cards)
        })
    
    except Exception as e:
        logger.error(f"Error getting audio cards: {e}")
        abort(500, f"Failed to get audio cards: {str(e)}")


@app.route("/spl/measure", methods=["GET"])
def measure_spl_level():
    """Measure SPL level using the specified or auto-detected microphone."""
    try:
        device = request.args.get("device")
        duration_str = request.args.get("duration", "1.0")
        duration = validate_float_param("duration", duration_str, 0.1, 10.0)
        
        result = measure_spl(device=device, duration=duration)
        
        if not result['success']:
            abort(500, result['error'])
        
        # Calculate effective sensitivity for display
        effective_sensitivity = None
        if result['microphone_sensitivity'] and result['microphone_gain'] is not None:
            effective_sensitivity = result['microphone_sensitivity'] - result['microphone_gain']
        
        return jsonify({
            "spl_db": result['spl_db'],
            "rms_db_fs": result['rms_db_fs'],
            "device": result['device'],
            "duration": result['duration'],
            "microphone": {
                "sensitivity": result['microphone_sensitivity'],
                "gain_db": result['microphone_gain'],
                "effective_sensitivity": effective_sensitivity
            },
            "timestamp": time.time(),
            "success": result['success']
        })
    
    except Exception as e:
        logger.error(f"Error measuring SPL: {e}")
        abort(500, f"Failed to measure SPL: {str(e)}")


def _monitor_playback():
    """Background thread to monitor playback and stop when time expires."""
    global _signal_generator, _playback_state
    
    while _playback_state['active']:
        time.sleep(0.1)
        current_time = datetime.now()
        
        if _playback_state['stop_time'] and current_time >= _playback_state['stop_time']:
            logger.info("Playback time expired, stopping...")
            if _signal_generator:
                _signal_generator.stop()
            _playback_state['active'] = False
            _playback_state['stop_time'] = None
            break


@app.route("/audio/analyze/fft", methods=["POST"])
def analyze_fft():
    """Perform FFT analysis on a WAV file."""
    # Get parameters
    filename = request.args.get("filename")
    filepath = request.args.get("filepath")
    window_type = request.args.get("window", "hann")
    fft_size_str = request.args.get("fft_size")
    start_time_str = request.args.get("start_time", "0")
    duration_str = request.args.get("duration")
    normalize_str = request.args.get("normalize")
    
    # Validate parameters
    if not filename and not filepath:
        abort(400, "Either 'filename' or 'filepath' parameter is required")
    
    if filename and filepath:
        abort(400, "Specify either 'filename' or 'filepath', not both")
    
    # Determine file path
    if filename:
        # Security validation for recorded files
        validated_path = validate_recording_file(filename)
        target_file = validated_path
    else:
        # External file path - basic security checks
        if not filepath.endswith('.wav'):
            abort(400, "Only WAV files are supported")
        if not os.path.exists(filepath):
            abort(404, "File not found")
        target_file = filepath
    
    # Validate optional parameters
    start_time = 0.0
    if start_time_str:
        start_time = validate_float_param("start_time", start_time_str, 0.0)
    
    duration = None
    if duration_str:
        duration = validate_float_param("duration", duration_str, 0.1, 300.0)
    
    fft_size = None
    if fft_size_str:
        try:
            fft_size = int(fft_size_str)
            if fft_size < 64 or fft_size > 65536:
                abort(400, "fft_size must be between 64 and 65536")
            # Ensure it's a power of 2
            if fft_size & (fft_size - 1) != 0:
                abort(400, "fft_size must be a power of 2")
        except ValueError:
            abort(400, "Invalid fft_size: must be an integer")
    
    normalize = None
    if normalize_str:
        normalize = validate_float_param("normalize", normalize_str, 0.1, 50000.0)
    
    try:
        # Validate parameters using FFT module
        validate_fft_parameters(fft_size, window_type)
        
        # Use FFT module for analysis
        result = analyze_wav_file(target_file, window_type, fft_size, start_time, duration, normalize)
        
        # Prepare response
        response = {
            "status": "success",
            "file_info": result["file_info"],
            "fft_analysis": result["fft_analysis"],
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"FFT analysis completed for {os.path.basename(target_file)}: "
                   f"{result['file_info']['analyzed_samples']} samples, "
                   f"{result['fft_analysis']['fft_size']} FFT size")
        
        return jsonify(response)
        
    except ValueError as e:
        logger.error(f"FFT analysis parameter error: {e}")
        abort(400, str(e))
    except RuntimeError as e:
        logger.error(f"FFT analysis runtime error: {e}")
        abort(500, str(e))
    except Exception as e:
        logger.error(f"FFT analysis error: {e}")
        abort(500, f"FFT analysis failed: {str(e)}")


@app.route("/audio/analyze/fft-recording/<recording_id>", methods=["POST"])
def analyze_fft_recording(recording_id: str):
    """Perform FFT analysis on a recorded file by recording ID."""
    
    logger.info(f"FFT analysis requested for recording {recording_id}")
    
    # Get recording status using the recording module
    recording_status = get_recording_status(recording_id)
    if recording_status is None:
        logger.warning(f"Recording {recording_id} not found - may have been cleaned up or never existed")
        abort(404, f"Recording {recording_id} not found. It may have been cleaned up or never existed.")
    
    if recording_status["status"] != "completed":
        logger.warning(f"Recording {recording_id} is not completed (status: {recording_status['status']})")
        abort(400, f"Recording {recording_id} is not completed. Current status: {recording_status['status']}")
    
    filepath = recording_status["filepath"]
    if not os.path.exists(filepath):
        logger.error(f"Recording file {filepath} no longer exists on disk")
        abort(404, "Recording file no longer available - may have been cleaned up")
    
    try:
        # Get analysis parameters
        window_type = request.args.get("window", "hann")
        fft_size_str = request.args.get("fft_size")
        start_time_str = request.args.get("start_time", "0")
        duration_str = request.args.get("duration")
        normalize_str = request.args.get("normalize")
        
        logger.debug(f"FFT analysis parameters for {recording_id}: window={window_type}, "
                    f"fft_size={fft_size_str}, start_time={start_time_str}, "
                    f"duration={duration_str}, normalize={normalize_str}")
        
        # Validate parameters
        start_time = 0.0
        if start_time_str:
            start_time = validate_float_param("start_time", start_time_str, 0.0)
        
        duration = None
        if duration_str:
            duration = validate_float_param("duration", duration_str, 0.1, 300.0)
        
        fft_size = None
        if fft_size_str:
            try:
                fft_size = int(fft_size_str)
                if fft_size < 64 or fft_size > 65536:
                    abort(400, "fft_size must be between 64 and 65536")
                if fft_size & (fft_size - 1) != 0:
                    abort(400, "fft_size must be a power of 2")
            except ValueError:
                abort(400, "Invalid fft_size: must be an integer")
        
        normalize = None
        if normalize_str:
            normalize = validate_float_param("normalize", normalize_str, 0.1, 50000.0)
        
        # Validate parameters using FFT module
        validate_fft_parameters(fft_size, window_type)
        
        logger.info(f"Starting FFT analysis for recording {recording_id} on file {filepath}")
        
        # Use FFT module for analysis
        result = analyze_wav_file(filepath, window_type, fft_size, start_time, duration, normalize)
        
        # Prepare response with recording context
        response = {
            "status": "success",
            "recording_info": {
                "recording_id": recording_id,
                "filename": recording_status["filename"],
                "original_duration": recording_status["duration"],
                "original_device": recording_status["device"],
                "original_sample_rate": recording_status["sample_rate"],
                "timestamp": recording_status["timestamp"]
            },
            "analysis_info": {
                "analyzed_duration": result["file_info"]["analyzed_duration"],
                "analyzed_samples": result["file_info"]["analyzed_samples"],
                "start_time": start_time
            },
            "fft_analysis": result["fft_analysis"],
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"FFT analysis completed for recording {recording_id}: "
                   f"{result['file_info']['analyzed_samples']} samples, "
                   f"{result['fft_analysis']['fft_size']} FFT size")
        
        return jsonify(response)
        
    except ValueError as e:
        logger.error(f"FFT analysis parameter error for recording {recording_id}: {e}")
        abort(400, str(e))
    except RuntimeError as e:
        logger.error(f"FFT analysis runtime error for recording {recording_id}: {e}")
        abort(500, str(e))
    except Exception as e:
        logger.error(f"FFT analysis error for recording {recording_id}: {e}")
        abort(500, f"FFT analysis failed: {str(e)}")


@app.route("/audio/record/start", methods=["POST"])
def start_recording_endpoint():
    """Start recording audio to a WAV file in background."""
    
    try:
        duration_str = request.args.get("duration", "10.0")
        device = request.args.get("device")
        sample_rate_str = request.args.get("sample_rate", "48000")
        
        duration = validate_float_param("duration", duration_str, 1.0, 300.0)  # Max 5 minutes
        
        try:
            sample_rate = int(sample_rate_str)
            if sample_rate not in [8000, 16000, 22050, 44100, 48000, 96000]:
                abort(400, "sample_rate must be one of: 8000, 16000, 22050, 44100, 48000, 96000")
        except ValueError:
            abort(400, "Invalid sample_rate: must be an integer")
        
        # Auto-detect device if not specified
        if device is None:
            try:
                detector = MicrophoneDetector()
                microphones = detector.detect_microphones()
                if not microphones:
                    abort(500, "No microphone detected")
                card_id = microphones[0][0]
                device = f"hw:{card_id},0"
            except Exception as e:
                abort(500, f"Failed to detect microphone: {str(e)}")
        
        # Generate unique recording ID
        recording_id = str(uuid.uuid4())[:8]  # Short UUID for easier handling
        
        # Start recording using the recording module
        start_recording(recording_id, device, duration, sample_rate)
        
        logger.info(f"Started recording {recording_id}: {duration}s on {device} at {sample_rate}Hz")
        
        return jsonify({
            "status": "started",
            "recording_id": recording_id,
            "filename": f"recording_{recording_id}.wav",
            "duration": duration,
            "device": device,
            "sample_rate": sample_rate,
            "estimated_completion": (datetime.now() + timedelta(seconds=duration)).isoformat(),
            "message": f"Recording started: {duration}s at {sample_rate}Hz"
        })
        
    except Exception as e:
        logger.error(f"Error starting recording: {e}")
        abort(500, f"Failed to start recording: {str(e)}")


@app.route("/audio/record/status/<recording_id>", methods=["GET"])
def get_recording_status_endpoint(recording_id: str):
    """Get the status of a specific recording."""
    
    status = get_recording_status(recording_id)
    if status is None:
        abort(404, f"Recording {recording_id} not found")
    
    # Add some additional fields for backwards compatibility
    if status["status"] == "active":
        elapsed = (datetime.now() - datetime.fromisoformat(status["start_time"])).total_seconds()
        remaining = max(0, status["duration"] - elapsed)
        status.update({
            "recording_id": recording_id,
            "elapsed_seconds": round(elapsed, 1),
            "remaining_seconds": round(remaining, 1),
            "completed": False
        })
    elif status["status"] == "completed":
        status.update({
            "recording_id": recording_id,
            "completed": True,
            "file_available": os.path.exists(status["filepath"])
        })
    
    return jsonify(status)


@app.route("/audio/record/list", methods=["GET"])
def list_recordings_endpoint():
    """List all recordings (active and completed)."""
    
    recordings_data = list_recordings()
    # Add temp directory for backwards compatibility
    recordings_data["temp_directory"] = recording_manager.get_temp_dir()
    
    return jsonify(recordings_data)


@app.route("/audio/record/download/<recording_id>", methods=["GET"])
def download_recording_endpoint(recording_id: str):
    """Download a completed recording file."""
    
    status = get_recording_status(recording_id)
    if status is None or status["status"] != "completed":
        abort(404, f"Recording {recording_id} not found or not completed")
    
    filepath = status["filepath"]
    
    if not os.path.exists(filepath):
        abort(404, "Recording file no longer available")
    
    return send_file(
        filepath,
        as_attachment=True,
        download_name=status["filename"],
        mimetype="audio/wav"
    )


@app.route("/audio/record/delete/<recording_id>", methods=["DELETE"])
def delete_recording_endpoint(recording_id: str):
    """Delete a specific recording file."""
    
    status = get_recording_status(recording_id)
    if status is None or status["status"] != "completed":
        abort(404, f"Recording {recording_id} not found or not completed")
    
    try:
        if delete_recording(recording_id):
            return jsonify({
                "status": "deleted",
                "recording_id": recording_id,
                "filename": status["filename"],
                "message": f"Recording {recording_id} deleted successfully"
            })
        else:
            abort(500, f"Failed to delete recording {recording_id}")
            
    except Exception as e:
        logger.error(f"Error deleting recording {recording_id}: {e}")
        abort(500, f"Failed to delete recording: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error deleting recording {recording_id}: {e}")
        abort(500, f"Failed to delete recording: {str(e)}")


@app.route("/audio/record/delete-file/<filename>", methods=["DELETE"])
def delete_recording_file_endpoint(filename: str):
    """Delete a recording file by filename (for security, only allows files in temp directory)."""
    try:
        if delete_recording_file(filename):
            return jsonify({
                "status": "deleted",
                "filename": filename,
                "message": f"Recording file {filename} deleted successfully"
            })
        else:
            abort(500, f"Failed to delete recording file {filename}")
        
    except Exception as e:
        logger.error(f"Error deleting recording file {filename}: {e}")
        abort(500, f"Failed to delete recording file: {str(e)}")
        logger.error(f"Error deleting recording file {filename}: {e}")
        abort(500, f"Failed to delete recording file: {str(e)}")


@app.route("/audio/record/cleanup", methods=["POST"])
def cleanup_recordings_endpoint():
    """Manually trigger cleanup of recordings older than 10 minutes."""
    try:
        # Get count before cleanup
        recordings_data = list_recordings()
        before_count = recordings_data["completed_count"]
        
        # Run cleanup
        cleanup_old_recordings()
        
        # Get count after cleanup
        recordings_data = list_recordings()
        after_count = recordings_data["completed_count"]
        removed_count = before_count - after_count
        
        return jsonify({
            "status": "success",
            "message": f"Cleanup completed successfully",
            "recordings_before": before_count,
            "recordings_after": after_count,
            "recordings_removed": removed_count
        })
        
    except Exception as e:
        logger.error(f"Error during manual cleanup: {e}")
        abort(500, f"Cleanup failed: {str(e)}")


@app.route("/audio/record/cleanup/status", methods=["GET"])
def cleanup_status_endpoint():
    """Get current cleanup status and settings."""
    
    return jsonify(get_cleanup_status())


@app.route("/audio/sweep/start", methods=["POST"])
def start_sine_sweep():
    """Start playing a sine sweep for the specified duration."""
    global _signal_generator, _playback_state
    
    try:
        duration_str = request.args.get("duration", "5.0")
        amplitude_str = request.args.get("amplitude", "0.5")
        start_freq_str = request.args.get("start_freq", "20")
        end_freq_str = request.args.get("end_freq", "20000")
        sweeps_str = request.args.get("sweeps", "1")
        device = request.args.get("device")
        
        duration = validate_float_param("duration", duration_str, 1.0, 30.0)
        amplitude = validate_float_param("amplitude", amplitude_str, 0.0, 1.0)
        start_freq = validate_float_param("start_freq", start_freq_str, 10.0, 22000.0)
        end_freq = validate_float_param("end_freq", end_freq_str, 10.0, 22000.0)
        
        try:
            sweeps = int(sweeps_str)
            if sweeps < 1 or sweeps > 10:
                abort(400, "sweeps must be between 1 and 10")
        except ValueError:
            abort(400, "Invalid sweeps: must be an integer")
        
        if start_freq >= end_freq:
            abort(400, "start_freq must be less than end_freq")
        
        # Calculate total duration
        total_duration = duration * sweeps
        
        # Stop any existing playback
        if _signal_generator and _playback_state['active']:
            _signal_generator.stop()
            _playback_state['active'] = False
        
        # Create new generator
        _signal_generator = SignalGenerator(device=device)
        
        # Set initial stop time based on total duration
        stop_time = datetime.now() + timedelta(seconds=total_duration)
        
        # Update playback state
        _playback_state.update({
            'active': True,
            'stop_time': stop_time,
            'amplitude': amplitude,
            'device': device,
            'signal_type': 'sine_sweep',
            'start_freq': start_freq,
            'end_freq': end_freq,
            'sweeps': sweeps,
            'sweep_duration': duration,
            'total_duration': total_duration
        })
        
        # Start playing multiple sine sweeps
        _signal_generator.play_sine_sweep(
            start_freq=start_freq,
            end_freq=end_freq,
            duration=duration,
            amplitude=amplitude,
            repeats=sweeps
        )
        
        # Start monitor thread
        monitor_thread = threading.Thread(target=_monitor_playback, daemon=True)
        monitor_thread.start()
        
        logger.info(f"Started {sweeps} sine sweep(s): {start_freq} Hz → {end_freq} Hz, {duration}s each, total {total_duration}s at {amplitude*100:.0f}% amplitude")
        
        return jsonify({
            "status": "started",
            "signal_type": "sine_sweep",
            "start_freq": start_freq,
            "end_freq": end_freq,
            "duration": duration,
            "sweeps": sweeps,
            "total_duration": total_duration,
            "amplitude": amplitude,
            "device": device or "default",
            "stop_time": stop_time.isoformat(),
            "message": f"{sweeps} sine sweep(s) started: {start_freq} Hz → {end_freq} Hz, {duration}s each (total: {total_duration}s)"
        })
        
    except Exception as e:
        logger.error(f"Error starting sine sweep: {e}")
        abort(500, f"Failed to start sine sweep: {str(e)}")


@app.route("/audio/noise/start", methods=["POST"])
def start_noise():
    """Start playing white noise for the specified duration."""
    global _signal_generator, _playback_state
    
    try:
        duration_str = request.args.get("duration", "3.0")
        amplitude_str = request.args.get("amplitude", "0.5")
        device = request.args.get("device")
        
        duration = validate_float_param("duration", duration_str, 1.0, 30.0)
        amplitude = validate_float_param("amplitude", amplitude_str, 0.0, 1.0)
        
        # Stop any existing playback
        if _signal_generator and _playback_state['active']:
            _signal_generator.stop()
            _playback_state['active'] = False
        
        # Create new generator
        _signal_generator = SignalGenerator(device=device)
        
        # Set initial stop time
        stop_time = datetime.now() + timedelta(seconds=duration)
        
        # Update playback state
        _playback_state.update({
            'active': True,
            'stop_time': stop_time,
            'amplitude': amplitude,
            'device': device,
            'signal_type': 'noise',
            'start_freq': None,
            'end_freq': None,
            'sweeps': None,
            'sweep_duration': None,
            'total_duration': None
        })
        
        # Start playing noise (infinite duration, will be stopped by monitor)
        _signal_generator.play_noise(duration=0, amplitude=amplitude)
        
        # Start monitor thread
        monitor_thread = threading.Thread(target=_monitor_playback, daemon=True)
        monitor_thread.start()
        
        logger.info(f"Started noise playback for {duration} seconds at {amplitude*100:.0f}% amplitude")
        
        return jsonify({
            "status": "started",
            "duration": duration,
            "amplitude": amplitude,
            "device": device or "default",
            "stop_time": stop_time.isoformat(),
            "message": f"Noise playback started for {duration} seconds"
        })
        
    except Exception as e:
        logger.error(f"Error starting noise: {e}")
        abort(500, f"Failed to start noise: {str(e)}")


@app.route("/audio/noise/keep-playing", methods=["POST"])
def keep_playing_noise():
    """Extend the current noise playback by the specified duration."""
    global _playback_state
    
    if not _playback_state['active']:
        abort(404, "No active noise playback to extend")
    
    try:
        duration_str = request.args.get("duration", "3.0")
        duration = validate_float_param("duration", duration_str, 1.0, 30.0)
        
        # Extend the stop time
        new_stop_time = datetime.now() + timedelta(seconds=duration)
        _playback_state['stop_time'] = new_stop_time
        
        logger.info(f"Extended noise playback by {duration} seconds")
        
        return jsonify({
            "status": "extended",
            "duration": duration,
            "new_stop_time": new_stop_time.isoformat(),
            "message": f"Playback extended by {duration} seconds"
        })
        
    except Exception as e:
        logger.error(f"Error extending playback: {e}")
        abort(500, f"Failed to extend playback: {str(e)}")


@app.route("/audio/noise/stop", methods=["POST"])
def stop_noise():
    """Stop the current noise playback immediately."""
    global _signal_generator, _playback_state
    
    try:
        if _signal_generator and _playback_state['active']:
            _signal_generator.stop()
            _playback_state['active'] = False
            _playback_state['stop_time'] = None
            
            logger.info("Stopped noise playback")
            
            return jsonify({
                "status": "stopped",
                "message": "Noise playback stopped"
            })
        else:
            return jsonify({
                "status": "not_active",
                "message": "No active noise playback to stop"
            })
            
    except Exception as e:
        logger.error(f"Error stopping noise: {e}")
        abort(500, f"Failed to stop noise: {str(e)}")


@app.route("/audio/noise/status", methods=["GET"])
def get_noise_status():
    """Get the current status of audio playback (noise or sine sweep)."""
    global _playback_state
    
    if _playback_state['active'] and _playback_state['stop_time']:
        remaining_time = (_playback_state['stop_time'] - datetime.now()).total_seconds()
        remaining_time = max(0, remaining_time)  # Don't show negative time
    else:
        remaining_time = 0
    
    status = {
        "active": _playback_state['active'],
        "signal_type": _playback_state['signal_type'],
        "amplitude": _playback_state['amplitude'],
        "device": _playback_state['device'] or "default",
        "remaining_seconds": round(remaining_time, 1),
        "stop_time": _playback_state['stop_time'].isoformat() if _playback_state['stop_time'] else None
    }
    
    # Add frequency information for sine sweeps
    if _playback_state['signal_type'] == 'sine_sweep':
        status.update({
            "start_freq": _playback_state['start_freq'],
            "end_freq": _playback_state['end_freq'],
            "sweeps": _playback_state['sweeps'],
            "sweep_duration": _playback_state['sweep_duration'],
            "total_duration": _playback_state['total_duration']
        })
    
    return jsonify(status)


@app.route("/", methods=["GET"])
def root():
    """Root endpoint with comprehensive API information."""
    return jsonify({
        "message": "RoomEQ Audio Processing API",
        "version": "0.3.0",
        "framework": "Flask",
        "description": "REST API for microphone detection, SPL measurement, audio signal generation, recording, and FFT analysis for acoustic measurements and room equalization",
        "features": [
            "Automatic microphone detection with sensitivity and gain information",
            "Real-time SPL (Sound Pressure Level) measurement",
            "White noise generation with keep-alive control",
            "Logarithmic sine sweep generation with multiple repeat support",
            "Background audio recording to WAV files with secure file management",
            "FFT spectral analysis with dB output and frequency normalization",
            "Windowing functions support (Hann, Hamming, Blackman)",
            "Frequency band analysis and automatic peak detection",
            "Real-time playback and recording management",
            "Cross-Origin Resource Sharing (CORS) support for web applications"
        ],
        "endpoints": {
            "info": {
                "/": "API information and documentation",
                "/version": "API version and feature list"
            },
            "microphones": {
                "/microphones": "Detect microphones with calibration data",
                "/microphones/raw": "Raw microphone detection (bash script compatible)"
            },
            "audio_devices": {
                "/audio/inputs": "List audio input card indices",
                "/audio/cards": "List all available audio cards"
            },
            "measurements": {
                "/spl/measure": "Measure sound pressure level and RMS"
            },
            "signal_generation": {
                "/audio/noise/start": "Start white noise playback",
                "/audio/noise/keep-playing": "Extend current noise playback",
                "/audio/noise/stop": "Stop current playback immediately",
                "/audio/noise/status": "Get current playback status",
                "/audio/sweep/start": "Start sine sweep(s) with multiple repeat support"
            },
            "recording": {
                "/audio/record/start": "Start recording audio to WAV file in background",
                "/audio/record/status/<recording_id>": "Get status of specific recording",
                "/audio/record/list": "List all recordings (active and completed)",
                "/audio/record/download/<recording_id>": "Download completed recording",
                "/audio/record/delete/<recording_id>": "Delete specific recording",
                "/audio/record/delete-file/<filename>": "Delete recording file by name"
            }
        },
        "usage_examples": {
            "microphone_detection": {
                "description": "Detect available microphones with calibration information",
                "method": "GET",
                "url": "/microphones",
                "example": "curl -X GET http://localhost:10315/microphones"
            },
            "spl_measurement": {
                "description": "Measure sound pressure level using detected microphone",
                "method": "GET", 
                "url": "/spl/measure",
                "parameters": {
                    "device": "ALSA device (optional, auto-detects if not specified)",
                    "duration": "Measurement duration in seconds (0.1-10.0, default: 1.0)"
                },
                "example": "curl -X GET 'http://localhost:10315/spl/measure?duration=2.0'"
            },
            "white_noise": {
                "description": "Generate white noise for acoustic testing",
                "method": "POST",
                "url": "/audio/noise/start", 
                "parameters": {
                    "duration": "Playback duration in seconds (1.0-30.0, default: 3.0)",
                    "amplitude": "Amplitude level (0.0-1.0, default: 0.5)",
                    "device": "Output device (optional, auto-detects if not specified)"
                },
                "example": "curl -X POST 'http://localhost:10315/audio/noise/start?duration=5&amplitude=0.3'"
            },
            "sine_sweep_single": {
                "description": "Generate single logarithmic sine sweep",
                "method": "POST",
                "url": "/audio/sweep/start",
                "parameters": {
                    "start_freq": "Starting frequency in Hz (10-22000, default: 20)",
                    "end_freq": "Ending frequency in Hz (10-22000, default: 20000)", 
                    "duration": "Sweep duration in seconds (1.0-30.0, default: 5.0)",
                    "amplitude": "Amplitude level (0.0-1.0, default: 0.5)",
                    "device": "Output device (optional, auto-detects if not specified)"
                },
                "example": "curl -X POST 'http://localhost:10315/audio/sweep/start?start_freq=20&end_freq=20000&duration=10&amplitude=0.4'"
            },
            "sine_sweep_multiple": {
                "description": "Generate multiple consecutive sine sweeps for averaging",
                "method": "POST",
                "url": "/audio/sweep/start",
                "parameters": {
                    "start_freq": "Starting frequency in Hz (10-22000, default: 20)",
                    "end_freq": "Ending frequency in Hz (10-22000, default: 20000)",
                    "duration": "Duration per sweep in seconds (1.0-30.0, default: 5.0)", 
                    "sweeps": "Number of consecutive sweeps (1-10, default: 1)",
                    "amplitude": "Amplitude level (0.0-1.0, default: 0.5)",
                    "device": "Output device (optional, auto-detects if not specified)"
                },
                "example": "curl -X POST 'http://localhost:10315/audio/sweep/start?start_freq=20&end_freq=20000&duration=8&sweeps=3&amplitude=0.3'"
            },
            "keep_alive": {
                "description": "Extend current playback to prevent automatic stop",
                "method": "POST",
                "url": "/audio/noise/keep-playing",
                "parameters": {
                    "duration": "Additional duration in seconds (1.0-30.0, default: 3.0)"
                },
                "example": "curl -X POST 'http://localhost:10315/audio/noise/keep-playing?duration=5'"
            },
            "audio_recording": {
                "description": "Record audio to WAV file in background",
                "method": "POST",
                "url": "/audio/record/start",
                "parameters": {
                    "duration": "Recording duration in seconds (1.0-300.0, default: 10.0)",
                    "device": "Input device (optional, auto-detects if not specified)",
                    "sample_rate": "Sample rate in Hz (8000|16000|22050|44100|48000|96000, default: 48000)"
                },
                "example": "curl -X POST 'http://localhost:10315/audio/record/start?duration=30&sample_rate=48000'"
            },
            "recording_management": {
                "list_recordings": "curl -X GET http://localhost:10315/audio/record/list",
                "check_status": "curl -X GET http://localhost:10315/audio/record/status/abc12345",
                "download": "curl -X GET http://localhost:10315/audio/record/download/abc12345 -o recording.wav",
                "delete_by_id": "curl -X DELETE http://localhost:10315/audio/record/delete/abc12345",
                "delete_by_filename": "curl -X DELETE http://localhost:10315/audio/record/delete-file/recording_abc12345.wav"
            },
            "playback_control": {
                "stop": "curl -X POST http://localhost:10315/audio/noise/stop",
                "status": "curl -X GET http://localhost:10315/audio/noise/status"
            }
        },
        "response_formats": {
            "microphones": {
                "description": "Array of microphone objects with calibration data",
                "example": [
                    {
                        "card_index": 1,
                        "device_name": "HiFiBerry Mic",
                        "sensitivity": -37.0,
                        "sensitivity_str": "-37",
                        "gain_db": 20
                    }
                ]
            },
            "spl_measurement": {
                "description": "SPL measurement with microphone calibration applied", 
                "example": {
                    "spl_db": 45.2,
                    "rms_db_fs": -42.1,
                    "device": "hw:1,0",
                    "duration": 1.0,
                    "microphone": {
                        "sensitivity": -37.0,
                        "gain_db": 20,
                        "effective_sensitivity": -57.0
                    },
                    "timestamp": 1692123456.789,
                    "success": True
                }
            },
            "playback_status": {
                "description": "Current playback status with detailed information",
                "example": {
                    "active": True,
                    "signal_type": "sine_sweep",
                    "amplitude": 0.3,
                    "device": "default", 
                    "remaining_seconds": 4.2,
                    "stop_time": "2025-08-15T12:15:30.123456",
                    "start_freq": 20.0,
                    "end_freq": 20000.0,
                    "sweeps": 3,
                    "sweep_duration": 8.0,
                    "total_duration": 24.0
                }
            },
            "recording_status": {
                "description": "Recording status and file information",
                "active_example": {
                    "recording_id": "abc12345",
                    "status": "recording",
                    "filename": "recording_abc12345.wav",
                    "device": "hw:1,0",
                    "duration": 30.0,
                    "sample_rate": 48000,
                    "elapsed_seconds": 12.5,
                    "remaining_seconds": 17.5,
                    "completed": False
                },
                "completed_example": {
                    "recording_id": "abc12345",
                    "status": "completed",
                    "filename": "recording_abc12345.wav",
                    "device": "hw:1,0",
                    "duration": 30.0,
                    "sample_rate": 48000,
                    "timestamp": "2025-08-15T12:20:00.123456",
                    "completed": True,
                    "file_available": True
                }
            }
        },
        "technical_details": {
            "audio_format": "16-bit PCM, 48kHz sample rate",
            "sweep_type": "Logarithmic frequency progression",
            "noise_type": "White noise with uniform frequency distribution", 
            "microphone_calibration": "Automatic sensitivity and gain compensation",
            "device_detection": "ALSA-based audio device enumeration",
            "playback_monitoring": "Real-time status tracking with automatic timeout",
            "cors_support": "Cross-origin requests enabled for web applications"
        },
        "server_info": {
            "host": "0.0.0.0",
            "port": 10315,
            "framework": "Flask with CORS support",
            "threading": "Multi-threaded request handling",
            "audio_backend": "ALSA with arecord fallback for compatibility"
        }
    })


def main():
    """Main entry point for the roomeq-server console script."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, recording_signal_handler)
    signal.signal(signal.SIGTERM, recording_signal_handler)
    
    # Start the cleanup timer
    start_cleanup_timer()
    
    app.run(
        host="0.0.0.0", 
        port=10315,
        debug=False,
        threaded=True
    )
