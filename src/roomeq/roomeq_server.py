#!/usr/bin/env python3
from flask import Flask, jsonify, request, abort, send_file
from flask_cors import CORS
from typing import List, Dict, Any, Optional
import logging
import time
import os
import threading
import numpy as np
import wave
import signal
import sys
import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path

# Local imports
from .fft_utils import fft_diff
from .recording import (
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
from .rust_optimizer import optimize_with_rust_json, RustOptimizerError
from .presets import list_target_curves, list_optimizer_presets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
FFT_BACKEND = os.environ.get('ROOMEQ_FFT_BACKEND', 'rust').lower()  # 'python' or 'rust'

# Import FFT backend based on configuration
if FFT_BACKEND == 'rust':
    try:
        from .fft_rust import load_wav_file, compute_fft, analyze_wav_file, validate_fft_parameters, is_rust_backend_available
        if not is_rust_backend_available():
            logger.warning("Rust FFT backend not available, falling back to Python")
            from .fft import load_wav_file, compute_fft, analyze_wav_file, validate_fft_parameters
            FFT_BACKEND = 'python'
        else:
            logger.info("Using Rust FFT backend")
    except ImportError as e:
        logger.warning(f"Failed to import Rust FFT backend: {e}, falling back to Python")
        from .fft import load_wav_file, compute_fft, analyze_wav_file, validate_fft_parameters
        FFT_BACKEND = 'python'
else:
    from .fft import load_wav_file, compute_fft, analyze_wav_file, validate_fft_parameters
    logger.info("Using Python FFT backend")

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
    response.headers["server"] = "roomeq-api/0.5.0"
    return response

# Add request logging middleware
@app.before_request
def log_request_info():
    """Log detailed request information for debugging."""
    logger.info(f"Request: {request.method} {request.url}")
    if request.args:
        logger.debug(f"Query parameters: {dict(request.args)}")
    if request.content_type:
        logger.debug(f"Content-Type: {request.content_type}")
    if request.headers:
        logger.debug(f"Headers: {dict(request.headers)}")
    try:
        if request.json:
            logger.debug(f"JSON payload: {request.json}")
    except Exception as e:
        logger.debug(f"Error parsing JSON: {e}")

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
    logger.warning(f"Request details - Content-Type: {request.content_type}, Content-Length: {request.content_length}")
    if request.data:
        logger.warning(f"Raw request data: {request.data[:200]}...")  # First 200 bytes
    return jsonify({
        "error": "Bad Request", 
        "message": error.description or "The request was invalid",
        "endpoint": request.endpoint,
        "url": request.url,
        "method": request.method
    }), 400

@app.errorhandler(500)
def internal_error(error):
    """Enhanced 500 error handler with detailed logging."""
    logger.error(f"500 Internal Server Error: {request.method} {request.url} - {error}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    return jsonify({
        "error": "Internal Server Error",
        "message": "An internal server error occurred",
        "endpoint": request.endpoint,
        "url": request.url,
        "method": request.method
    }), 500


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
        "version": "0.7.0",
        "api_name": "RoomEQ Audio Processing API",
        "features": [
            "Microphone detection with sensitivity and gain",
            "SPL measurement",
            "FFT analysis with windowing, normalization, and logarithmic frequency summarization",
            "Audio recording with automatic cleanup",
            "Sine sweep generation",
            "White/pink noise generation",
            "High-performance Rust-based room EQ optimization with real-time streaming",
            "Frequency deduplication and intelligent candidate generation",
            "Adaptive high-pass filter placement",
            "Usable frequency range detection",
            "Parametric EQ filter generation (biquad coefficients)",
            "Multiple optimizer presets with different smoothing characteristics",
            "Real-time streaming optimization progress (no buffering)"
        ],
        "server_info": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "flask_version": "2.x",
            "threading": "Multi-threaded request handling",
            "audio_backend": "ALSA with arecord fallback for compatibility",
            "fft_backend": f"{FFT_BACKEND.title()} FFT implementation",
            "optimization_backend": "High-performance Rust optimizer with streaming output",
            "rust_optimizer": "v0.6.0 with brute-force search and intelligent frequency management"
        }
    })


@app.route("/debug/routes", methods=["GET"])
def debug_routes():
    """Debug endpoint to show all available routes."""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": list(rule.methods - {"HEAD", "OPTIONS"}),
            "path": rule.rule
        })
    
    return jsonify({
        "available_routes": sorted(routes, key=lambda x: x["path"]),
        "total_routes": len(routes)
    })


@app.route("/config/fft-backend", methods=["GET"])
def get_fft_backend():
    """Get current FFT backend configuration."""
    backend_info = {
        "current_backend": FFT_BACKEND,
        "available_backends": ["python", "rust"],
        "configuration_source": "Environment variable ROOMEQ_FFT_BACKEND or default"
    }
    
    # Check if Rust backend is actually available
    if FFT_BACKEND == "rust":
        try:
            from .fft_rust import is_rust_backend_available
            backend_info["rust_available"] = is_rust_backend_available()
        except ImportError:
            backend_info["rust_available"] = False
    else:
        backend_info["rust_available"] = None
    
    return jsonify(backend_info)


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
    start_at_str = request.args.get("start_at")  # Alternative name for start_time
    duration_str = request.args.get("duration")
    normalize_str = request.args.get("normalize")
    points_per_octave_str = request.args.get("points_per_octave")
    
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
    # Handle both start_time and start_at (start_at takes precedence if both provided)
    if start_at_str and start_time_str and start_time_str != "0":
        abort(400, "Specify either 'start_time' or 'start_at', not both")
    
    start_time = 0.0
    if start_at_str:
        start_time = validate_float_param("start_at", start_at_str, 0.0)
    elif start_time_str:
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
    
    points_per_octave = None
    if points_per_octave_str:
        try:
            points_per_octave = int(points_per_octave_str)
            if points_per_octave < 1 or points_per_octave > 100:
                abort(400, "points_per_octave must be between 1 and 100")
        except ValueError:
            abort(400, "Invalid points_per_octave: must be an integer")
    
    
    try:
        # Validate parameters using FFT module
        validate_fft_parameters(fft_size, window_type)
        
        # Use FFT module for analysis
        result = analyze_wav_file(target_file, window_type, fft_size, start_time, duration, normalize, 
                                points_per_octave)
        
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
        start_at_str = request.args.get("start_at")  # Alternative name for start_time
        duration_str = request.args.get("duration")
        normalize_str = request.args.get("normalize")
        points_per_octave_str = request.args.get("points_per_octave")
        
        logger.debug(f"FFT analysis parameters for {recording_id}: window={window_type}, "
                    f"fft_size={fft_size_str}, start_time={start_time_str}, start_at={start_at_str}, "
                    f"duration={duration_str}, normalize={normalize_str}, "
                    f"points_per_octave={points_per_octave_str}")
        
        # Validate parameters
        # Handle both start_time and start_at (start_at takes precedence if both provided)
        if start_at_str and start_time_str and start_time_str != "0":
            abort(400, "Specify either 'start_time' or 'start_at', not both")
        
        start_time = 0.0
        if start_at_str:
            start_time = validate_float_param("start_at", start_at_str, 0.0)
        elif start_time_str:
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
        
        points_per_octave = None
        if points_per_octave_str:
            try:
                points_per_octave = int(points_per_octave_str)
                if points_per_octave < 1 or points_per_octave > 100:
                    abort(400, "points_per_octave must be between 1 and 100")
            except ValueError:
                abort(400, "Invalid points_per_octave: must be an integer")
        
        
        # Validate parameters using FFT module
        validate_fft_parameters(fft_size, window_type)
        
        logger.info(f"Starting FFT analysis for recording {recording_id} on file {filepath}")
        
        # Use FFT module for analysis
        result = analyze_wav_file(filepath, window_type, fft_size, start_time, duration, normalize, 
                                points_per_octave)
        
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


@app.route("/audio/analyze/fft-diff", methods=["POST"])
def analyze_fft_diff():
    """Perform FFT difference analysis between two recordings."""
    
    try:
        # Get request parameters - support both recording IDs and filenames
        recording_id1 = request.args.get("recording_id1")
        recording_id2 = request.args.get("recording_id2")
        filename1 = request.args.get("filename1") 
        filename2 = request.args.get("filename2")
        filepath1 = request.args.get("filepath1")
        filepath2 = request.args.get("filepath2")
        
        # Check that we have exactly one way to specify each file
        file1_methods = sum(bool(x) for x in [recording_id1, filename1, filepath1])
        file2_methods = sum(bool(x) for x in [recording_id2, filename2, filepath2])
        
        if file1_methods != 1 or file2_methods != 1:
            abort(400, "Must specify exactly one of recording_id1/filename1/filepath1 and recording_id2/filename2/filepath2")
        
        # Validate that we're not comparing the same file
        if (recording_id1 and recording_id2 and recording_id1 == recording_id2) or \
           (filename1 and filename2 and filename1 == filename2) or \
           (filepath1 and filepath2 and filepath1 == filepath2):
            abort(400, "Cannot compare the same file with itself")
        
        # Get optional FFT parameters
        window_type = request.args.get("window", "hann")
        fft_size_str = request.args.get("fft_size")
        start_time_str = request.args.get("start_time", "0")
        start_at_str = request.args.get("start_at")
        duration_str = request.args.get("duration")
        normalize_str = request.args.get("normalize")
        points_per_octave_str = request.args.get("points_per_octave")
        normalize_frequency_str = request.args.get("normalize_frequency", "1000")  # Default to 1kHz
        
        # Handle both start_time and start_at (start_at takes precedence if both provided)
        if start_at_str and start_time_str and start_time_str != "0":
            logger.warning("Both start_time and start_at provided. Using start_at and ignoring start_time")
        
        start_time = 0.0
        if start_at_str:
            start_time = validate_float_param("start_at", start_at_str, 0.0, 300.0)
        elif start_time_str:
            start_time = validate_float_param("start_time", start_time_str, 0.0, 300.0)
        
        duration = None
        if duration_str:
            duration = validate_float_param("duration", duration_str, 0.1, 300.0)
        
        fft_size = None
        if fft_size_str:
            try:
                fft_size = int(fft_size_str)
                if fft_size < 64 or fft_size > 65536 or (fft_size & (fft_size - 1)) != 0:
                    abort(400, "fft_size must be a power of 2 between 64 and 65536")
            except ValueError:
                abort(400, "Invalid fft_size: must be an integer")
        
        normalize = None
        if normalize_str:
            normalize = validate_float_param("normalize", normalize_str, 0.1, 50000.0)
        
        points_per_octave = None
        if points_per_octave_str:
            try:
                points_per_octave = int(points_per_octave_str)
                if points_per_octave < 1 or points_per_octave > 100:
                    abort(400, "points_per_octave must be between 1 and 100")
            except ValueError:
                abort(400, "Invalid points_per_octave: must be an integer")
        
        # Parse normalize_frequency parameter
        normalize_frequency = None
        if normalize_frequency_str and normalize_frequency_str.lower() != "none":
            normalize_frequency = validate_float_param("normalize_frequency", normalize_frequency_str, 10.0, 22000.0)
        
        
        
        logger.info(f"FFT difference analysis requested between files")
        
        # Determine file paths and metadata for each file
        file1_path = None
        file1_info = {"source_type": None, "identifier": None, "filename": None}
        file2_path = None 
        file2_info = {"source_type": None, "identifier": None, "filename": None}
        
        # Handle first file
        if recording_id1:
            recording_status1 = get_recording_status(recording_id1)
            if recording_status1 is None:
                abort(404, f"Recording {recording_id1} not found")
            if recording_status1["status"] != "completed":
                abort(400, f"Recording {recording_id1} is not completed")
            file1_path = recording_status1["filepath"]
            file1_info = {
                "source_type": "recording_id",
                "identifier": recording_id1,
                "filename": recording_status1["filename"],
                "timestamp": recording_status1.get("timestamp")
            }
        elif filename1:
            # Look for filename using comprehensive file resolution
            file1_path, file1_info = _resolve_audio_file(filename1)
            if file1_path is None:
                # Build helpful error message showing where we looked
                error_msg = f"File '{os.path.basename(filename1)}' not found. Searched in: "
                search_locations = []
                
                # Check where we searched
                signals_dir = "/usr/share/hifiberry/signals"
                if not os.path.exists(signals_dir):
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                    dev_signals_dir = os.path.join(project_root, "debian/roomeq/usr/share/hifiberry/signals")
                    if os.path.exists(dev_signals_dir):
                        signals_dir = dev_signals_dir
                
                search_locations.append(f"pre-created signals ({signals_dir})")
                search_locations.append("recording manager")
                search_locations.append("/tmp directory")
                
                error_msg += ", ".join(search_locations)
                abort(404, error_msg)
        else:  # filepath1
            if not os.path.exists(filepath1):
                abort(404, f"File '{filepath1}' not found")
            file1_path = filepath1
            file1_info = {
                "source_type": "filepath",
                "identifier": filepath1,
                "filename": os.path.basename(filepath1)
            }
        
        # Handle second file
        if recording_id2:
            recording_status2 = get_recording_status(recording_id2)
            if recording_status2 is None:
                abort(404, f"Recording {recording_id2} not found")
            if recording_status2["status"] != "completed":
                abort(400, f"Recording {recording_id2} is not completed")
            file2_path = recording_status2["filepath"]
            file2_info = {
                "source_type": "recording_id",
                "identifier": recording_id2,
                "filename": recording_status2["filename"],
                "timestamp": recording_status2.get("timestamp")
            }
        elif filename2:
            # Look for filename using comprehensive file resolution
            file2_path, file2_info = _resolve_audio_file(filename2)
            if file2_path is None:
                # Build helpful error message showing where we looked
                error_msg = f"File '{os.path.basename(filename2)}' not found. Searched in: "
                search_locations = []
                
                # Check where we searched
                signals_dir = "/usr/share/hifiberry/signals"
                if not os.path.exists(signals_dir):
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                    dev_signals_dir = os.path.join(project_root, "debian/roomeq/usr/share/hifiberry/signals")
                    if os.path.exists(dev_signals_dir):
                        signals_dir = dev_signals_dir
                
                search_locations.append(f"pre-created signals ({signals_dir})")
                search_locations.append("recording manager")
                search_locations.append("/tmp directory")
                
                error_msg += ", ".join(search_locations)
                abort(404, error_msg)
        else:  # filepath2
            if not os.path.exists(filepath2):
                abort(404, f"File '{filepath2}' not found")
            file2_path = filepath2
            file2_info = {
                "source_type": "filepath",
                "identifier": filepath2,
                "filename": os.path.basename(filepath2)
            }
        
        # Final validation that files exist
        if not os.path.exists(file1_path):
            abort(404, f"File 1 no longer exists: {file1_path}")
        if not os.path.exists(file2_path):
            abort(404, f"File 2 no longer exists: {file2_path}")
        
        # Validate parameters using FFT module
        validate_fft_parameters(fft_size, window_type)
        
        logger.info(f"Analyzing file 1 ({file1_info['source_type']}): {file1_path}")
        
        # Analyze first file
        result1 = analyze_wav_file(file1_path, window_type, fft_size, start_time, duration, normalize, 
                                 points_per_octave)
        
        logger.info(f"Analyzing file 2 ({file2_info['source_type']}): {file2_path}")
        
        # Analyze second file
        result2 = analyze_wav_file(file2_path, window_type, fft_size, start_time, duration, normalize, 
                                 points_per_octave)
        
        logger.info(f"Computing FFT difference between files")
        
        # Compute difference using fft_utils
        # If smoothing is requested (points_per_octave specified), prefer smoothed data
        prefer_smoothed = points_per_octave is not None
        diff_result = fft_diff(result1, result2, 
                              title=f"Difference: {file1_info['filename']} vs {file2_info['filename']}",
                              description=f"FFT difference analysis between {file1_info['filename']} and {file2_info['filename']}",
                              prefer_smoothed=prefer_smoothed,
                              normalize_frequency=normalize_frequency)
        
        # Prepare comprehensive response
        response = {
            "status": "success",
            "comparison_info": {
                "file1": {
                    "source_type": file1_info["source_type"],
                    "identifier": file1_info["identifier"],
                    "filename": file1_info["filename"],
                    "timestamp": file1_info.get("timestamp")
                },
                "file2": {
                    "source_type": file2_info["source_type"],
                    "identifier": file2_info["identifier"], 
                    "filename": file2_info["filename"],
                    "timestamp": file2_info.get("timestamp")
                },
                "analysis_parameters": {
                    "window_type": window_type,
                    "fft_size": result1["fft_analysis"]["fft_size"],
                    "points_per_octave": points_per_octave,
                    "normalize": normalize,
                    "normalize_frequency": normalize_frequency,
                    "start_time": start_time,
                    "analyzed_duration": result1["file_info"]["analyzed_duration"],
                    "analyzed_samples": result1["file_info"]["analyzed_samples"]
                }
            },
            "difference_analysis": diff_result,
            "individual_analyses": {
                "file1_fft": {
                    "peak_frequency": result1["fft_analysis"]["peak_frequency"],
                    "peak_magnitude": result1["fft_analysis"]["peak_magnitude"],
                    "spectral_centroid": result1["fft_analysis"]["spectral_centroid"]
                },
                "file2_fft": {
                    "peak_frequency": result2["fft_analysis"]["peak_frequency"],
                    "peak_magnitude": result2["fft_analysis"]["peak_magnitude"],
                    "spectral_centroid": result2["fft_analysis"]["spectral_centroid"]
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"FFT difference analysis completed between {file1_info['filename']} and {file2_info['filename']}: "
                   f"RMS difference: {diff_result['statistics']['rms_difference_db']:.2f} dB, "
                   f"Max difference: {diff_result['statistics']['max_difference_db']:.2f} dB")
        
        return jsonify(response)
        
    except ValueError as e:
        logger.error(f"FFT difference analysis parameter error: {e}")
        abort(400, str(e))
    except RuntimeError as e:
        logger.error(f"FFT difference analysis runtime error: {e}")
        abort(500, str(e))
    except Exception as e:
        logger.error(f"FFT difference analysis error: {e}")
        abort(500, f"FFT difference analysis failed: {str(e)}")


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
                    logger.error("No microphone detected for recording")
                    return jsonify({
                        "error": "No microphone detected",
                        "message": "No microphone was detected. Please ensure a microphone is connected and accessible.",
                        "suggestions": [
                            "Check if microphone is physically connected",
                            "Verify ALSA configuration",
                            "Check microphone permissions"
                        ]
                    }), 500
                card_id = microphones[0][0]
                device = f"hw:{card_id},0"
                logger.info(f"Auto-detected microphone device: {device}")
            except Exception as e:
                logger.error(f"Error during microphone detection: {e}")
                return jsonify({
                    "error": "Microphone detection failed",
                    "message": f"Failed to detect microphone: {str(e)}",
                    "suggestions": [
                        "Check system audio configuration",
                        "Verify ALSA is properly installed",
                        "Check microphone hardware connection"
                    ]
                }), 500
        
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
        return jsonify({
            "error": "Recording failed",
            "message": f"Failed to start recording: {str(e)}",
            "recording_id": recording_id if 'recording_id' in locals() else None
        }), 500


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



# =============================================================================
# Pre-created Signals API - List and access bundled signal files
# =============================================================================

@app.route("/audio/signals/list", methods=["GET"])
def list_signals():
    """List available pre-created signal files."""
    try:
        # Standard location for pre-created signals
        signals_dir = "/usr/share/hifiberry/signals"
        
        # Fallback to development location if standard location doesn't exist
        if not os.path.exists(signals_dir):
            # Try relative to the project root for development
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            dev_signals_dir = os.path.join(project_root, "debian/roomeq/usr/share/hifiberry/signals")
            if os.path.exists(dev_signals_dir):
                signals_dir = dev_signals_dir
            else:
                return jsonify({
                    "signals": [],
                    "signals_directory": signals_dir,
                    "available": False,
                    "message": "No pre-created signals directory found"
                })
        
        signals = []
        
        if os.path.exists(signals_dir):
            for filename in os.listdir(signals_dir):
                if filename.lower().endswith('.wav'):
                    filepath = os.path.join(signals_dir, filename)
                    
                    # Get file info
                    try:
                        import wave
                        with wave.open(filepath, 'rb') as wf:
                            duration = wf.getnframes() / wf.getframerate()
                            channels = wf.getnchannels()
                            sample_rate = wf.getframerate()
                            file_size = os.path.getsize(filepath)
                    except Exception:
                        # If we can't read WAV info, provide basic file info
                        duration = None
                        channels = None
                        sample_rate = None
                        file_size = os.path.getsize(filepath)
                    
                    # Parse signal info from filename
                    signal_info = _parse_signal_filename(filename)
                    
                    signal_entry = {
                        "filename": filename,
                        "filepath": filepath,
                        "file_size": file_size,
                        "duration": duration,
                        "channels": channels,
                        "sample_rate": sample_rate,
                        **signal_info
                    }
                    
                    signals.append(signal_entry)
        
        # Sort by filename for consistent ordering
        signals.sort(key=lambda x: x['filename'])
        
        logger.info(f"Listed {len(signals)} pre-created signals from {signals_dir}")
        
        return jsonify({
            "signals": signals,
            "count": len(signals),
            "signals_directory": signals_dir,
            "available": True,
            "message": f"Found {len(signals)} pre-created signal(s)"
        })
        
    except Exception as e:
        logger.error(f"Error listing signals: {e}")
        return jsonify({
            "signals": [],
            "count": 0,
            "signals_directory": signals_dir if 'signals_dir' in locals() else "unknown",
            "available": False,
            "error": str(e),
            "message": "Failed to list signals"
        }), 500


def _parse_signal_filename(filename):
    """Parse signal information from filename."""
    # Remove extension
    name = filename.lower().replace('.wav', '')
    
    signal_info = {
        "signal_type": "unknown",
        "description": filename
    }
    
    # Parse sweep signals: sweep_10hz_22000hz_10s
    if name.startswith('sweep_'):
        parts = name.split('_')
        if len(parts) >= 4:
            try:
                start_freq_str = parts[1].replace('hz', '')
                end_freq_str = parts[2].replace('hz', '')
                duration_str = parts[3].replace('s', '')
                
                start_freq = float(start_freq_str)
                end_freq = float(end_freq_str)
                duration = float(duration_str)
                
                signal_info.update({
                    "signal_type": "sine_sweep",
                    "start_freq": start_freq,
                    "end_freq": end_freq,
                    "sweep_duration": duration,
                    "description": f"Sine sweep {start_freq} Hz → {end_freq} Hz, {duration}s"
                })
            except ValueError:
                pass
    
    # Parse noise signals: noise_white_5s, noise_pink_10s  
    elif name.startswith('noise_'):
        parts = name.split('_')
        if len(parts) >= 3:
            try:
                noise_type = parts[1]  # white, pink, etc.
                duration_str = parts[2].replace('s', '')
                duration = float(duration_str)
                
                signal_info.update({
                    "signal_type": "noise",
                    "noise_type": noise_type,
                    "duration": duration,
                    "description": f"{noise_type.title()} noise, {duration}s"
                })
            except ValueError:
                pass
    
    return signal_info


def _resolve_audio_file(filename):
    """
    Resolve an audio filename to a full path by searching in multiple locations.
    
    Search order:
    1. Pre-created signals directory (/usr/share/hifiberry/signals or dev location)
    2. Recording manager (completed recordings and generated files)
    3. /tmp directory (for generated files)
    
    Returns:
        tuple: (file_path, file_info) where file_info contains metadata about the source
    """
    search_filename = os.path.basename(filename)
    
    # 1. Check pre-created signals directory first
    signals_dir = "/usr/share/hifiberry/signals"
    
    # Fallback to development location if standard location doesn't exist
    if not os.path.exists(signals_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        dev_signals_dir = os.path.join(project_root, "debian/roomeq/usr/share/hifiberry/signals")
        if os.path.exists(dev_signals_dir):
            signals_dir = dev_signals_dir
    
    if os.path.exists(signals_dir):
        signals_path = os.path.join(signals_dir, search_filename)
        if os.path.exists(signals_path):
            logger.info(f"Found pre-created signal: {signals_path}")
            return signals_path, {
                "source_type": "pre_created_signal",
                "identifier": filename,
                "filename": search_filename,
                "signals_directory": signals_dir,
                "file_type": "pre_created_signal"
            }
    
    # 2. Check recording manager (includes generated files and recordings)
    try:
        from .recording import recording_manager
        recording_files = recording_manager.get_completed_recordings()
        for rec_id, rec_data in recording_files.items():
            if rec_data["filename"] == search_filename:
                logger.info(f"Found in recording manager: {rec_data['filepath']}")
                return rec_data["filepath"], {
                    "source_type": "recording_manager",
                    "identifier": filename,
                    "filename": search_filename,
                    "timestamp": rec_data.get("timestamp"),
                    "file_type": rec_data.get("file_type", "recording"),
                    "recording_id": rec_id
                }
    except Exception as e:
        logger.warning(f"Error checking recording manager for {search_filename}: {e}")
    
    # 3. Check /tmp directory (fallback for generated files)
    tmp_path = os.path.join('/tmp', search_filename)
    if os.path.exists(tmp_path):
        logger.info(f"Found in /tmp: {tmp_path}")
        return tmp_path, {
            "source_type": "temp_directory",
            "identifier": filename,
            "filename": search_filename,
            "file_type": "generated"
        }
    
    # File not found in any location
    return None, None


# =============================================================================
# Generator API - Generate audio files without immediate playback
# =============================================================================

@app.route("/audio/generate/sweep", methods=["POST"])
def generate_sweep():
    """Generate a sine sweep file without playing it."""
    try:
        duration_str = request.args.get("duration", "5.0")
        amplitude_str = request.args.get("amplitude", "0.5")
        start_freq_str = request.args.get("start_freq", "20")
        end_freq_str = request.args.get("end_freq", "20000")
        sweeps_str = request.args.get("sweeps", "1")
        compensation_mode = request.args.get("compensation_mode", "none").lower()
        
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
            
        # Validate compensation mode
        valid_modes = {"none", "inv_sqrt_f", "sqrt_f"}
        if compensation_mode not in valid_modes:
            abort(400, f"compensation_mode must be one of {sorted(valid_modes)}")

        # Calculate total duration
        total_duration = duration * sweeps
        
        # Create generator (no device needed for file generation)
        generator = SignalGenerator()
        
        # Generate sweep file
        sweep_file = generator.generate_sweep_file(
            start_freq=start_freq,
            end_freq=end_freq,
            duration=duration,
            amplitude=amplitude,
            compensation_mode=compensation_mode,
            sweeps=sweeps
        )
        
        logger.info(f"Generated sweep file: {os.path.basename(sweep_file)} - {sweeps} sweep(s), {start_freq} Hz → {end_freq} Hz, {duration}s each")
        
        return jsonify({
            "status": "generated",
            "signal_type": "sine_sweep", 
            "filename": os.path.basename(sweep_file),
            "filepath": sweep_file,
            "start_freq": start_freq,
            "end_freq": end_freq,
            "duration": duration,
            "sweeps": sweeps,
            "total_duration": total_duration,
            "amplitude": amplitude,
            "compensation_mode": compensation_mode,
            "message": f"Generated {sweeps} sine sweep(s): {start_freq} Hz → {end_freq} Hz, {duration}s each (total: {total_duration}s)"
        })
        
    except Exception as e:
        logger.error(f"Error generating sweep: {e}")
        abort(500, f"Failed to generate sweep: {str(e)}")


@app.route("/audio/generate/noise", methods=["POST"])
def generate_noise():
    """Generate a white noise file without playing it."""
    try:
        duration_str = request.args.get("duration", "3.0")
        amplitude_str = request.args.get("amplitude", "0.5")
        
        duration = validate_float_param("duration", duration_str, 1.0, 60.0)  # Max 60 seconds
        amplitude = validate_float_param("amplitude", amplitude_str, 0.0, 1.0)
        
        # Create generator (no device needed for file generation)
        generator = SignalGenerator()
        
        # Generate noise file
        noise_file = generator.generate_noise_file(duration=duration, amplitude=amplitude)
        
        logger.info(f"Generated noise file: {os.path.basename(noise_file)} - {duration}s at {amplitude*100:.0f}% amplitude")
        
        return jsonify({
            "status": "generated",
            "signal_type": "noise",
            "filename": os.path.basename(noise_file),
            "filepath": noise_file,
            "duration": duration,
            "amplitude": amplitude,
            "message": f"Generated white noise: {duration}s at {amplitude*100:.0f}% amplitude"
        })
        
    except Exception as e:
        logger.error(f"Error generating noise: {e}")
        abort(500, f"Failed to generate noise: {str(e)}")


# =============================================================================
# Player API - Play back generated or existing audio files
# =============================================================================

@app.route("/audio/play/file", methods=["POST"])
def play_file():
    """Play an audio file."""
    global _signal_generator, _playback_state
    
    try:
        filepath = request.args.get("filepath") or request.args.get("filename")
        device = request.args.get("device")
        repeats_str = request.args.get("repeats", "1")
        
        if not filepath:
            abort(400, "filepath or filename parameter is required")
        
        # Validate repeats parameter
        try:
            repeats = int(repeats_str)
            if repeats < 1 or repeats > 100:  # Reasonable upper limit
                abort(400, "repeats must be between 1 and 100")
        except ValueError:
            abort(400, "Invalid repeats: must be an integer")
        
        # If only a filename is provided, use comprehensive file resolution
        if not os.path.isabs(filepath):
            resolved_path, file_info = _resolve_audio_file(filepath)
            if resolved_path is None:
                # Build helpful error message
                error_msg = f"Audio file not found: {filepath}. Searched in: "
                search_locations = []
                
                signals_dir = "/usr/share/hifiberry/signals"
                if not os.path.exists(signals_dir):
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                    dev_signals_dir = os.path.join(project_root, "debian/roomeq/usr/share/hifiberry/signals")
                    if os.path.exists(dev_signals_dir):
                        signals_dir = dev_signals_dir
                
                search_locations.append(f"pre-created signals ({signals_dir})")
                search_locations.append("recording manager")
                search_locations.append("/tmp directory")
                
                error_msg += ", ".join(search_locations)
                abort(404, error_msg)
            
            filepath = resolved_path
            logger.info(f"Using {file_info['source_type']}: {filepath}")
        
        if not os.path.exists(filepath):
            abort(404, f"Audio file not found: {filepath}")
        
        # Stop any existing playback
        if _signal_generator and _playback_state['active']:
            _signal_generator.stop()
            _playback_state['active'] = False
        
        # Create new generator
        _signal_generator = SignalGenerator(device=device)
        
        # Get file info for tracking
        try:
            import wave
            with wave.open(filepath, 'rb') as wf:
                file_duration_single = wf.getnframes() / wf.getframerate()
        except Exception:
            # Fallback if we can't read the file info
            file_duration_single = 10.0  # Rough estimate
        
        # Calculate total duration including repeats and gaps
        total_duration = (file_duration_single * repeats) + (0.1 * max(0, repeats - 1))  # 0.1s gap between repeats
        
        # Set stop time based on total duration
        stop_time = datetime.now() + timedelta(seconds=total_duration)
        
        # Update playback state
        _playback_state.update({
            'active': True,
            'stop_time': stop_time,
            'amplitude': None,  # Not applicable for file playback
            'device': device,
            'signal_type': 'file',
            'start_freq': None,
            'end_freq': None,
            'sweeps': None,
            'sweep_duration': None,
            'total_duration': total_duration,
            'compensation_mode': None,
            'filename': os.path.basename(filepath),
            'filepath': filepath,
            'repeats': repeats,
            'file_duration_single': file_duration_single
        })
        
        # Start file playback with repeats
        started = _signal_generator.play_file(filepath, repeats=repeats)
        
        if not started:
            abort(500, "Failed to start file playback")
            
        # Wait briefly to ensure playback started
        for _ in range(10):
            if _signal_generator.is_playing():
                break
            time.sleep(0.05)
        
        # Start monitor thread
        monitor_thread = threading.Thread(target=_monitor_playback, daemon=True)
        monitor_thread.start()
        
        logger.info(f"Started playing file: {os.path.basename(filepath)} ({repeats} time{'s' if repeats != 1 else ''})")
        
        return jsonify({
            "status": "started",
            "signal_type": "file",
            "filename": os.path.basename(filepath),
            "filepath": filepath,
            "duration": file_duration_single,
            "repeats": repeats,
            "total_duration": total_duration,
            "device": device or "default",
            "stop_time": stop_time.isoformat(),
            "message": f"Started playing file: {os.path.basename(filepath)} ({repeats} time{'s' if repeats != 1 else ''})"
        })
        
    except Exception as e:
        logger.error(f"Error playing file: {e}")
        abort(500, f"Failed to play file: {str(e)}")


@app.route("/audio/play/stop", methods=["POST"])
def stop_playback():
    """Stop any current audio playback."""
    global _signal_generator, _playback_state
    
    try:
        if _signal_generator and _playback_state['active']:
            _signal_generator.stop()
            _playback_state['active'] = False
            _playback_state['stop_time'] = None
            
            logger.info("Stopped audio playback")
            
            return jsonify({
                "status": "stopped",
                "message": "Audio playback stopped"
            })
        else:
            return jsonify({
                "status": "not_active",
                "message": "No active playback to stop"
            })
            
    except Exception as e:
        logger.error(f"Error stopping playback: {e}")
        abort(500, f"Failed to stop playback: {str(e)}")


@app.route("/audio/play/status", methods=["GET"])
def get_playback_status():
    """Get the current status of audio playback."""
    global _playback_state
    
    if _playback_state['active'] and _playback_state['stop_time']:
        remaining_time = (_playback_state['stop_time'] - datetime.now()).total_seconds()
        remaining_time = max(0, remaining_time)  # Don't show negative time
    else:
        remaining_time = 0
    
    status = {
        "active": _playback_state['active'],
        "signal_type": _playback_state['signal_type'],
        "device": _playback_state['device'] or "default",
        "remaining_seconds": round(remaining_time, 1),
        "stop_time": _playback_state['stop_time'].isoformat() if _playback_state['stop_time'] else None,
        "filename": _playback_state.get('filename'),
        "filepath": _playback_state.get('filepath')
    }
    
    # Add signal-specific information
    if _playback_state['signal_type'] == 'sine_sweep':
        status.update({
            "start_freq": _playback_state['start_freq'],
            "end_freq": _playback_state['end_freq'],
            "sweeps": _playback_state['sweeps'],
            "sweep_duration": _playback_state['sweep_duration'],
            "compensation_mode": _playback_state['compensation_mode']
        })
    elif _playback_state['signal_type'] == 'file':
        status.update({
            "total_duration": _playback_state['total_duration'],
            "repeats": _playback_state.get('repeats', 1),
            "file_duration_single": _playback_state.get('file_duration_single')
        })
    elif _playback_state['signal_type'] in ['noise']:
        status.update({
            "total_duration": _playback_state['total_duration']
        })
    
    if _playback_state.get('amplitude') is not None:
        status["amplitude"] = _playback_state['amplitude']
    
    return jsonify(status)


# =============================================================================
# Legacy Combined Generator/Player API (for backwards compatibility)
# =============================================================================

@app.route("/audio/sweep/start", methods=["POST"])
def start_sine_sweep():
    """Start playing a sine sweep for the specified duration.
    
    DEPRECATED: Use /audio/generate/sweep + /audio/play/file for better separation of concerns.
    """
    global _signal_generator, _playback_state
    
    try:
        duration_str = request.args.get("duration", "5.0")
        amplitude_str = request.args.get("amplitude", "0.5")
        start_freq_str = request.args.get("start_freq", "20")
        end_freq_str = request.args.get("end_freq", "20000")
        sweeps_str = request.args.get("sweeps", "1")
        compensation_mode = request.args.get("compensation_mode", "none").lower()
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
        # Validate compensation mode
        valid_modes = {"none", "inv_sqrt_f", "sqrt_f"}
        if compensation_mode not in valid_modes:
            abort(400, f"compensation_mode must be one of {sorted(valid_modes)}")

        # Calculate total duration
        total_duration = duration * sweeps
        
        # Stop any existing playback
        if _signal_generator and _playback_state['active']:
            _signal_generator.stop()
            _playback_state['active'] = False
        
        # Create new generator
        _signal_generator = SignalGenerator(device=device)
        
        # Generate sweep file for tracking
        sweep_file = _signal_generator.generate_sweep_file(
            start_freq=start_freq,
            end_freq=end_freq,
            duration=duration,
            amplitude=amplitude,
            compensation_mode=compensation_mode,
            sweeps=sweeps
        )
        
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
            'total_duration': total_duration,
            'compensation_mode': compensation_mode,
            'filename': os.path.basename(sweep_file)
        })

        # Start playing multiple sine sweeps using native generator
        started = _signal_generator.play_sine_sweep(
            start_freq=start_freq,
            end_freq=end_freq,
            duration=duration,
            amplitude=amplitude,
            repeats=sweeps,
            compensation_mode=compensation_mode
        )

        # Ensure playback thread has started before returning
        if not started:
            abort(500, "Failed to start sweep playback")
        # Wait briefly to ensure PCM started; poll is_playing for up to ~0.5s
        for _ in range(10):
            if _signal_generator.is_playing():
                break
            time.sleep(0.05)
        
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
            "compensation_mode": compensation_mode,
            "device": device or "default",
            "stop_time": stop_time.isoformat(),
            "filename": os.path.basename(sweep_file),
            "message": f"{sweeps} sine sweep(s) started: {start_freq} Hz → {end_freq} Hz, {duration}s each (total: {total_duration}s)"
        })
        
    except Exception as e:
        logger.error(f"Error starting sine sweep: {e}")
        abort(500, f"Failed to start sine sweep: {str(e)}")


@app.route("/audio/noise/start", methods=["POST"])
def start_noise():
    """Start playing white noise for the specified duration.
    
    DEPRECATED: Use /audio/generate/noise + /audio/play/file for better separation of concerns.
    """
    global _signal_generator, _playback_state
    
    try:
        duration_str = request.args.get("duration", "3.0")
        amplitude_str = request.args.get("amplitude", "0.5")
        device = request.args.get("device")
        
        duration = validate_float_param("duration", duration_str, 1.0, 60.0)  # Max 60 seconds
        amplitude = validate_float_param("amplitude", amplitude_str, 0.0, 1.0)
        
        # Stop any existing playback
        if _signal_generator and _playback_state['active']:
            _signal_generator.stop()
            _playback_state['active'] = False
        
        # Create new generator
        _signal_generator = SignalGenerator(device=device)
        
        # Set stop time for monitoring
        stop_time = datetime.now() + timedelta(seconds=duration)
        
        # Generate noise file for playback
        noise_filename = _signal_generator.generate_noise_file(duration=duration, amplitude=amplitude)
        
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
            'total_duration': duration,
            'filename': noise_filename
        })
        
        # Start playing noise with the actual duration (no infinite duration)
        _signal_generator.play_noise(duration=duration, amplitude=amplitude)
        
        # Start monitor thread as backup safety measure
        monitor_thread = threading.Thread(target=_monitor_playback, daemon=True)
        monitor_thread.start()
        
        logger.info(f"Started noise playback for {duration} seconds at {amplitude*100:.0f}% amplitude")
        
        return jsonify({
            "status": "started",
            "duration": duration,
            "amplitude": amplitude,
            "device": device or "default",
            "filename": noise_filename,
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
    """Stop the current noise playback immediately.
    
    DEPRECATED: Use /audio/play/stop for unified playback control.
    """
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
    """Get the current status of audio playback (noise or sine sweep).
    
    DEPRECATED: Use /audio/play/status for unified playback status.
    """
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
        "stop_time": _playback_state['stop_time'].isoformat() if _playback_state['stop_time'] else None,
        "filename": _playback_state.get('filename')
    }
    
    # Add frequency information for sine sweeps
    if _playback_state['signal_type'] == 'sine_sweep':
        status.update({
            "start_freq": _playback_state['start_freq'],
            "end_freq": _playback_state['end_freq'],
            "sweeps": _playback_state['sweeps'],
            "sweep_duration": _playback_state['sweep_duration'],
            "total_duration": _playback_state['total_duration'],
            "compensation_mode": _playback_state.get('compensation_mode', 'sqrt_f')
        })
    
    return jsonify(status)


# EQ Optimizer endpoints
@app.route("/eq/presets/targets", methods=["GET"])
def eq_targets():
    """Get available EQ target curves."""
    try:
        curves = list_target_curves()
        return jsonify({
            "success": True,
            "target_curves": curves,
            "count": len(curves)
        })
    except Exception as e:
        logger.error(f"Error getting target curves: {e}")
        abort(500, f"Failed to get target curves: {str(e)}")


@app.route("/eq/presets/optimizers", methods=["GET"])
def eq_optimizers():
    """Get available EQ optimizer presets."""
    try:
        presets = list_optimizer_presets()
        return jsonify({
            "success": True,
            "optimizer_presets": presets,
            "count": len(presets)
        })
    except Exception as e:
        logger.error(f"Error getting optimizer presets: {e}")
        abort(500, f"Failed to get optimizer presets: {str(e)}")


@app.route("/eq/optimize", methods=["POST"])
def eq_optimize():
    """Run EQ optimization from complete JSON job definition with streaming output."""
    from flask import Response
    
    # Extract the complete job JSON from the request
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    job_data = request.get_json()
    
    def generate_optimization_stream():
        try:
            # Validate that we have a complete job definition
            if not job_data:
                yield f"data: {json.dumps({'error': 'No job data provided'})}\n\n"
                return
            
            # Send initial status
            yield f"data: {json.dumps({'type': 'started', 'message': 'Starting optimization', 'job_data': job_data})}\n\n"
            
            # Pass the complete JSON object directly to the Rust optimizer
            for step in optimize_with_rust_json(job_data):
                yield f"data: {json.dumps(step)}\n\n"
                
        except RustOptimizerError as e:
            logger.error(f"Rust optimizer error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        except Exception as e:
            logger.error(f"Error in optimization stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': f'Optimization failed: {str(e)}'})}\n\n"
    
    return Response(
        generate_optimization_stream(),
        mimetype='text/plain',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Content-Type-Options': 'nosniff'
        }
    )


@app.route("/eq/usable-range", methods=["POST"])
def eq_usable_range():
    """Calculate usable frequency range from measured curve data."""
    
    # Extract the request data
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    job_data = request.get_json()
    
    try:
        # Validate that we have at least measured_curve
        if not job_data or 'measured_curve' not in job_data:
            return jsonify({"error": "measured_curve is required"}), 400
        
        measured_curve = job_data['measured_curve']
        if 'frequencies' not in measured_curve or 'magnitudes_db' not in measured_curve:
            return jsonify({"error": "measured_curve must contain 'frequencies' and 'magnitudes_db'"}), 400
        
        # Import the Rust optimizer here to avoid circular import
        from .rust_optimizer import RustOptimizer
        
        # Create optimizer instance
        sample_rate = job_data.get('sample_rate', 48000.0)
        optimizer = RustOptimizer()
        
        # Detect usable frequency range using the Rust binary
        result = optimizer.detect_usable_range(job_data)
        
        return jsonify({
            "success": True,
            "usable_frequency_range": {
                "low_hz": result.usable_freq_low,
                "high_hz": result.usable_freq_high
            },
            "frequency_candidates": result.frequency_candidates,
            "optimization_frequencies": result.optimization_frequencies,
            "message": result.message
        })
        
    except RustOptimizerError as e:
        logger.error(f"Rust optimizer error in usable range detection: {e}")
        return jsonify({"error": f"Usable range detection failed: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Error in usable range detection: {e}")
        return jsonify({"error": f"Usable range detection failed: {str(e)}"}), 500


@app.route("/", methods=["GET"])
def root():
    """Root endpoint with comprehensive API information."""
    return jsonify({
        "message": "RoomEQ Audio Processing API",
        "version": "0.6.0",
        "framework": "Flask",
        "description": "REST API for microphone detection, SPL measurement, audio signal generation, recording, FFT analysis, difference analysis, and automatic room EQ optimization with real-time progress reporting",
        "features": [
            "Automatic microphone detection with sensitivity and gain information",
            "Real-time SPL (Sound Pressure Level) measurement",
            "White noise generation with keep-alive control and file tracking",
            "Logarithmic sine sweep generation with multiple repeat support",
            "Background audio recording to WAV files with secure file management",
            "FFT spectral analysis with dB output, frequency normalization, and logarithmic frequency summarization",
            "FFT difference analysis for comparing two recordings with detailed statistics",
            "Automatic room EQ optimization with multiple target curves and optimizer presets",
            "Real-time optimization progress reporting with step-by-step updates",
            "Biquad filter generation for parametric EQ implementation",
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
                "/audio/generate/sweep": "Generate sine sweep file",
                "/audio/generate/noise": "Generate white noise file",
                "/audio/play/file": "Play audio file with repeats support",
                "/audio/play/stop": "Stop audio playback",
                "/audio/play/status": "Get playback status",
                "/audio/signals/list": "List pre-created signals",
                "/audio/noise/start": "Start white noise playback (deprecated)",
                "/audio/noise/keep-playing": "Extend current noise playback (deprecated)",
                "/audio/noise/stop": "Stop current playback (deprecated)",
                "/audio/noise/status": "Get current playback status (deprecated)",
                "/audio/sweep/start": "Start sine sweep(s) (deprecated)"
            },
            "recording": {
                "/audio/record/start": "Start recording audio to WAV file in background",
                "/audio/record/status/<recording_id>": "Get status of specific recording",
                "/audio/record/list": "List all recordings (active and completed)",
                "/audio/record/download/<recording_id>": "Download completed recording",
                "/audio/record/delete/<recording_id>": "Delete specific recording",
                "/audio/record/delete-file/<filename>": "Delete recording file by name"
            },
            "fft_analysis": {
                "/audio/analyze/fft": "Analyze WAV file with FFT (by filename or filepath)",
                "/audio/analyze/fft-recording/<recording_id>": "Analyze recorded file with FFT by recording ID",
                "/audio/analyze/fft-diff": "Compare two recordings with FFT difference analysis"
            },
            "eq_optimization": {
                "/eq/presets/targets": "List available target response curves",
                "/eq/presets/optimizers": "List available optimizer configurations", 
                "/eq/optimize": "Run EQ optimization with streaming results (no start/stop needed)",
                "/eq/usable-range": "Calculate usable frequency range from measured curve data"
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
                "description": "Generate white noise for acoustic testing with file tracking",
                "method": "POST",
                "url": "/audio/noise/start", 
                "parameters": {
                    "duration": "Playback duration in seconds (1.0-60.0, default: 3.0)",
                    "amplitude": "Amplitude level (0.0-1.0, default: 0.5)",
                    "device": "Output device (optional, auto-detects if not specified)"
                },
                "response": "Includes filename of generated noise file for tracking",
                "example": "curl -X POST 'http://localhost:10315/audio/noise/start?duration=5&amplitude=0.3'",
                "response_example": {
                    "status": "started",
                    "duration": 5.0,
                    "amplitude": 0.3,
                    "device": "default",
                    "filename": "/tmp/roomeq_noise_abc123def456.wav",
                    "stop_time": "2025-08-20T12:05:00.000000",
                    "message": "Noise playback started for 5.0 seconds"
                }
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
                    "compensation_mode": "Amplitude compensation ('none' | 'inv_sqrt_f' | 'sqrt_f', default: 'sqrt_f')",
                    "device": "Output device (optional, auto-detects if not specified)"
                },
                "example": "curl -X POST 'http://localhost:10315/audio/sweep/start?start_freq=20&end_freq=20000&duration=10&amplitude=0.4&compensation_mode=none'",
                "response_example": {
                    "status": "started",
                    "signal_type": "sine_sweep",
                    "start_freq": 20,
                    "end_freq": 20000,
                    "duration": 10.0,
                    "sweeps": 1,
                    "total_duration": 10.0,
                    "amplitude": 0.4,
                    "compensation_mode": "none",
                    "device": "default",
                    "filename": "roomeq_sweep_abc123def456.wav",
                    "stop_time": "2025-08-20T12:05:00.000000",
                    "message": "1 sine sweep(s) started: 20 Hz → 20000 Hz, 10.0s each (total: 10.0s)"
                }
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
                    "compensation_mode": "Amplitude compensation ('none' | 'inv_sqrt_f' | 'sqrt_f', default: 'sqrt_f')",
                    "device": "Output device (optional, auto-detects if not specified)"
                },
                "example": "curl -X POST 'http://localhost:10315/audio/sweep/start?start_freq=20&end_freq=20000&duration=8&sweeps=3&amplitude=0.3&compensation_mode=none'",
                "response_example": {
                    "status": "started",
                    "signal_type": "sine_sweep",
                    "start_freq": 20,
                    "end_freq": 20000,
                    "duration": 8.0,
                    "sweeps": 3,
                    "total_duration": 24.0,
                    "amplitude": 0.3,
                    "compensation_mode": "none",
                    "device": "default",
                    "filename": "roomeq_sweep_def789abc012.wav",
                    "stop_time": "2025-08-20T12:05:24.000000",
                    "message": "3 sine sweep(s) started: 20 Hz → 20000 Hz, 8.0s each (total: 24.0s)"
                }
            },
            "sine_sweep_sox": {
                "description": "Generate sine sweep using SoX (creates a temp WAV in /tmp, then plays it)",
                "method": "POST",
                "url": "/audio/sweep/start",
                "parameters": {
                    "start_freq": "Starting frequency in Hz (10-22000, default: 20)",
                    "end_freq": "Ending frequency in Hz (10-22000, default: 20000)",
                    "duration": "Sweep duration in seconds (1.0-30.0, default: 5.0)",
                    "amplitude": "Amplitude level (0.0-1.0, default: 0.5)",
                    "sweeps": "Number of consecutive sweeps (1-10, default: 1)",
                    "device": "Output device (optional)"
                },
                "example": "curl -X POST 'http://localhost:10315/audio/sweep/start?start_freq=20&end_freq=20000&duration=8&sweeps=2&amplitude=0.3&generator=sine_sox'",
                "response_example": {
                    "status": "started",
                    "signal_type": "sine_sweep",
                    "start_freq": 20,
                    "end_freq": 20000,
                    "duration": 8.0,
                    "sweeps": 2,
                    "total_duration": 16.0,
                    "amplitude": 0.3,
                    "compensation_mode": "sqrt_f",
                    "device": "default",
                    "filename": "roomeq_sweep_fed654cba321.wav",
                    "stop_time": "2025-08-20T12:05:16.000000",
                    "message": "2 sine sweep(s) started: 20 Hz → 20000 Hz, 8.0s each (total: 16.0s)"
                }
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
            "fft_analysis": {
                "description": "Perform FFT analysis on WAV files with optional logarithmic frequency summarization",
                "method": "POST",
                "url": "/audio/analyze/fft",
                "parameters": {
                    "filename": "Recorded filename (use with recording system)",
                    "filepath": "External WAV file path (alternative to filename)",
                    "window": "Window function (hann|hamming|blackman|rectangular|none, default: hann)",
                    "fft_size": "FFT size, power of 2 (64-65536, default: auto)",
                    "start_time": "Start analysis at time in seconds (default: 0)",
                    "start_at": "Alternative to start_time - start analysis at time in seconds (default: 0)",
                    "duration": "Analyze duration in seconds (0.1-300, default: entire file)",
                    "normalize": "Normalize to frequency in Hz (0.1-50000, optional)",
                    "points_per_octave": "Summarize to log frequency buckets (1-100, optional)",
                },
                "examples": {
                    "basic": "curl -X POST 'http://localhost:10315/audio/analyze/fft?filename=recording_abc12345.wav'",
                    "with_start_at": "curl -X POST 'http://localhost:10315/audio/analyze/fft?filename=recording_abc12345.wav&start_at=2.5&duration=10'",
                    "with_summarization": "curl -X POST 'http://localhost:10315/audio/analyze/fft?filename=recording_abc12345.wav&points_per_octave=16'",
                    "external_file": "curl -X POST 'http://localhost:10315/audio/analyze/fft?filepath=/path/to/file.wav&window=hann&normalize=1000'",
                    "recording_analysis": "curl -X POST 'http://localhost:10315/audio/analyze/fft-recording/abc12345?points_per_octave=12&start_at=1.0'",
                    "difference_analysis": "curl -X POST 'http://localhost:10315/audio/analyze/fft-diff?recording_id1=abc12345&recording_id2=def67890&points_per_octave=16'"
                }
            },
            "fft_difference_analysis": {
                "description": "Compare two audio files using FFT difference analysis to identify signal processing effects. Supports multiple input methods: recording IDs, filenames, or file paths.",
                "method": "POST",
                "url": "/audio/analyze/fft-diff",
                "parameters": {
                    "recording_id1": "First recording ID (mutually exclusive with filename1/filepath1)",
                    "filename1": "First filename in recording directory (mutually exclusive with recording_id1/filepath1)",  
                    "filepath1": "Full path to first file (mutually exclusive with recording_id1/filename1)",
                    "recording_id2": "Second recording ID (mutually exclusive with filename2/filepath2)",
                    "filename2": "Second filename in recording directory (mutually exclusive with recording_id2/filepath2)",
                    "filepath2": "Full path to second file (mutually exclusive with recording_id2/filename2)",
                    "window": "Window function (hann|hamming|blackman|rectangular|none, default: hann)",
                    "fft_size": "FFT size, power of 2 (64-65536, default: auto)",
                    "start_time": "Start analysis at time in seconds (default: 0)",
                    "start_at": "Alternative to start_time - start analysis at time in seconds (default: 0)",
                    "duration": "Analyze duration in seconds (0.1-300, default: entire file)",
                    "normalize": "Normalize to frequency in Hz (0.1-50000, optional)",
                    "points_per_octave": "Summarize to log frequency buckets (1-100, optional)",
                },
                "examples": {
                    "recording_ids": "curl -X POST 'http://localhost:10315/audio/analyze/fft-diff?recording_id1=abc12345&recording_id2=def67890'",
                    "filenames": "curl -X POST 'http://localhost:10315/audio/analyze/fft-diff?filename1=noise_abc123.wav&filename2=recording_def456.wav'", 
                    "filepaths": "curl -X POST 'http://localhost:10315/audio/analyze/fft-diff?filepath1=/tmp/file1.wav&filepath2=/tmp/file2.wav'",
                    "mixed_types": "curl -X POST 'http://localhost:10315/audio/analyze/fft-diff?recording_id1=abc12345&filename2=noise_def456.wav'",
                    "with_options": "curl -X POST 'http://localhost:10315/audio/analyze/fft-diff?filename1=noise_signal.wav&filename2=room_recording.wav&points_per_octave=16'",
                    "time_segment": "curl -X POST 'http://localhost:10315/audio/analyze/fft-diff?recording_id1=abc12345&recording_id2=def67890&start_at=2.0&duration=5.0'"
                },
                "use_cases": [
                    "Compare noise playback file vs room recording to analyze room response", 
                    "Compare input signal vs processed output to measure system effects",
                    "Analyze before/after recordings to quantify acoustic treatment effectiveness",
                    "Measure loudspeaker performance differences between positions or settings",
                    "Compare generated noise files with recorded measurements"
                ]
            },
            "playback_control": {
                "stop": "curl -X POST http://localhost:10315/audio/noise/stop",
                "status": "curl -X GET http://localhost:10315/audio/noise/status",
                "status_note": "Status response includes filename for noise playback"
            },
            "eq_optimization": {
                "description": "Automatic room EQ optimization with real-time streaming progress",
                "list_targets": "curl -X GET http://localhost:10315/eq/presets/targets",
                "list_optimizers": "curl -X GET http://localhost:10315/eq/presets/optimizers",
                "optimize_from_recording": {
                    "method": "POST",
                    "url": "/eq/optimize",
                    "example": "curl -X POST http://localhost:10315/eq/optimize -H 'Content-Type: application/json' -d '{\"recording_id\":\"abc12345\",\"target_curve\":\"weighted_flat\",\"optimizer_preset\":\"default\",\"filter_count\":8}'",
                    "response": "Server-Sent Events stream with real-time progress and final results",
                    "parameters": {
                        "recording_id": "ID of completed recording to analyze",
                        "target_curve": "Target response curve (weighted_flat, flat, falling_slope, room_only, harman, vocal_presence)",
                        "optimizer_preset": "Optimization style (default, smooth, aggressive, verysmooth, precise)",
                        "filter_count": "Number of EQ filters to generate (1-20, default: 8)",
                        "window": "FFT window function for analysis (hann, hamming, blackman)",
                        "points_per_octave": "Frequency resolution for optimization (1-100, default: 12)"
                    }
                },
                "optimize_from_fft": {
                    "method": "POST", 
                    "url": "/eq/optimize",
                    "example": "curl -X POST http://localhost:10315/eq/optimize -H 'Content-Type: application/json' -d '{\"frequencies\":[20,25,31.5,...],\"magnitudes\":[-5.2,-3.1,-2.8,...],\"target_curve\":\"harman\",\"filter_count\":6}'",
                    "response": "Server-Sent Events stream with real-time progress and final results",
                    "parameters": {
                        "frequencies": "Array of frequency values in Hz",
                        "magnitudes": "Array of magnitude values in dB (same length as frequencies)",
                        "target_curve": "Target response curve name",
                        "optimizer_preset": "Optimization style name",
                        "filter_count": "Number of filters to generate",
                        "sample_rate": "Audio sample rate (default: 48000)"
                    }
                },
                "usable_frequency_range": {
                    "method": "POST",
                    "url": "/eq/usable-range", 
                    "example": "curl -X POST http://localhost:10315/eq/usable-range -H 'Content-Type: application/json' -d '{\"measured_curve\":{\"frequencies\":[20,25,31.5,...],\"magnitudes_db\":[-5.2,-3.1,-2.8,...]}}'",
                    "response": "JSON object with usable frequency range information",
                    "parameters": {
                        "measured_curve": "Required: Object with 'frequencies' and 'magnitudes_db' arrays",
                        "optimizer_params": "Optional: Object with min_frequency and max_frequency overrides",
                        "sample_rate": "Optional: Audio sample rate (default: 48000)"
                    }
                }
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
                "description": "Current playback status with detailed information including filename for noise",
                "noise_example": {
                    "active": True,
                    "signal_type": "noise",
                    "amplitude": 0.5,
                    "device": "default",
                    "remaining_seconds": 12.3,
                    "stop_time": "2025-08-20T12:15:30.123456",
                    "filename": "/tmp/roomeq_noise_abc123def456.wav"
                },
                "sweep_example": {
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
                    "total_duration": 24.0,
                    "compensation_mode": "sqrt_f",
                    "filename": None
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
            },
            "fft_analysis": {
                "description": "FFT analysis results with optional logarithmic frequency summarization",
                "standard_example": {
                    "status": "success",
                    "file_info": {
                        "filename": "recording_abc12345.wav",
                        "analyzed_duration": 10.0,
                        "analyzed_samples": 480000,
                        "start_time": 0.0
                    },
                    "fft_analysis": {
                        "fft_size": 32768,
                        "window_type": "hann",
                        "sample_rate": 48000,
                        "frequency_resolution": 1.46,
                        "frequencies": [0.0, 1.46, 2.93, "..."],
                        "magnitudes": [-80.2, -78.5, -76.1, "..."],
                        "phases": [0.0, 0.15, -0.32, "..."],
                        "peak_frequency": 1000.0,
                        "peak_magnitude": -20.5,
                        "spectral_centroid": 2500.0,
                        "frequency_bands": {
                            "bass": {"range": "60-250 Hz", "avg_magnitude": -45.2, "peak_frequency": 120.0},
                            "midrange": {"range": "500-2000 Hz", "avg_magnitude": -25.8, "peak_frequency": 1000.0}
                        },
                        "normalization": {"applied": False}
                    },
                    "timestamp": "2025-08-15T12:25:00.123456"
                },
                "summarized_example": {
                    "status": "success",
                    "file_info": {
                        "filename": "recording_abc12345.wav",
                        "analyzed_duration": 10.0,
                        "analyzed_samples": 480000,
                        "start_time": 0.0
                    },
                    "fft_analysis": {
                        "fft_size": 32768,
                        "window_type": "hann",
                        "sample_rate": 48000,
                        "frequency_resolution": 1.46,
                        "log_frequency_summary": {
                            "frequencies": [20.0, 23.8, 28.3, 33.7, 40.1, 47.8, "..."],
                            "magnitudes": [-65.2, -62.8, -58.4, -55.1, -52.3, -48.9, "..."],
                            "points_per_octave": 16,
                            "frequency_range": [20.0, 20000.0],
                            "n_octaves": 10.0,
                            "n_points": 161
                        },
                        "peak_frequency": 1000.0,
                        "peak_magnitude": -20.5,
                        "spectral_centroid": 2500.0,
                        "frequency_bands": {
                            "bass": {"range": "60-250 Hz", "avg_magnitude": -45.2, "peak_frequency": 120.0}
                        },
                        "normalization": {"applied": False}
                    },
                    "timestamp": "2025-08-15T12:25:00.123456"
                }
            },
            "fft_difference": {
                "description": "FFT difference analysis results with statistics and comparison data",
                "example": {
                    "status": "success",
                    "recording_info": {
                        "recording_id1": "abc12345",
                        "filename1": "recording_abc12345.wav",
                        "timestamp1": "2025-08-15T12:20:00.123456",
                        "recording_id2": "def67890",
                        "filename2": "recording_def67890.wav",
                        "timestamp2": "2025-08-15T12:25:00.123456"
                    },
                    "analysis_info": {
                        "analyzed_duration": 10.0,
                        "analyzed_samples": 480000,
                        "start_time": 0.0,
                        "analysis_parameters": {
                            "window_type": "hann",
                            "fft_size": 32768,
                            "points_per_octave": 16,
                            "normalize": None
                        }
                    },
                    "fft_difference": {
                        "title": "Difference: abc12345 vs def67890",
                        "description": "FFT difference analysis between recordings",
                        "diff_type": "magnitude_difference_db",
                        "sample_rate": 48000,
                        "frequencies": [20.0, 23.8, 28.3, 33.7, "..."],
                        "magnitudes": [-2.3, -1.8, -1.2, -0.8, "..."],
                        "phases": [0.0, 0.0, 0.0, 0.0, "..."],
                        "peak_frequency": 2500.0,
                        "peak_magnitude": -8.7,
                        "statistics": {
                            "rms_difference_db": 3.2,
                            "max_difference_db": 8.7,
                            "mean_difference_db": -1.1,
                            "frequency_range": [20.0, 20000.0],
                            "n_points": 161
                        },
                        "source_info": {
                            "result1_title": "Result 1",
                            "result2_title": "Result 2",
                            "result1_peak_freq": 1000.0,
                            "result2_peak_freq": 1200.0
                        },
                        "spectral_density": {
                            "type": "FFT Magnitude Difference",
                            "units": "dB difference",
                            "description": "Magnitude difference between two FFT results",
                            "computation": "result1_db - result2_db"
                        }
                    },
                    "analysis_timestamp": "2025-08-15T12:30:00.123456"
                }
            },
            "eq_optimization": {
                "target_curves": {
                    "description": "Available target response curves for EQ optimization",
                    "example": {
                        "flat": {
                            "name": "Flat Response",
                            "description": "Perfectly flat frequency response (0 dB across all frequencies)"
                        },
                        "weighted_flat": {
                            "name": "Weighted Flat",
                            "description": "Flat response with frequency weighting"
                        },
                        "harman": {
                            "name": "Harman Target Curve",
                            "description": "Research-based preferred room response curve"
                        }
                    }
                },
                "optimizer_presets": {
                    "description": "Available optimizer configurations with different trade-offs",
                    "example": {
                        "default": {
                            "name": "Default",
                            "description": "Balanced optimization with moderate smoothing",
                            "smoothing_factor": 1.0,
                            "max_gain_db": 12.0,
                            "min_q": 0.5,
                            "max_q": 10.0
                        },
                        "smooth": {
                            "name": "Smooth",
                            "description": "Gentle corrections with increased smoothing",
                            "smoothing_factor": 2.0,
                            "max_gain_db": 8.0
                        },
                        "aggressive": {
                            "name": "Aggressive",
                            "description": "Strong corrections with minimal smoothing",
                            "max_gain_db": 15.0,
                            "smoothing_factor": 0.5
                        }
                    }
                },
                "optimization_progress": {
                    "description": "Real-time optimization progress with detailed steps",
                    "example": {
                        "optimization_id": "opt_abc12345",
                        "status": "optimizing",
                        "progress": 65.0,
                        "current_step": "Optimizing filter 5/8",
                        "steps_completed": 13,
                        "total_steps": 20,
                        "elapsed_time": 12.5,
                        "estimated_remaining": 7.2,
                        "current_filter": {
                            "index": 5,
                            "frequency": 2500.0,
                            "gain_db": -3.8,
                            "q": 2.1,
                            "filter_type": "peaking_eq"
                        }
                    }
                },
                "optimization_streaming": {
                    "description": "Real-time Server-Sent Events stream for EQ optimization",
                    "content_type": "text/event-stream",
                    "progress_example": "data: {\"type\": \"progress\", \"message\": \"Optimization progress: 25/100 iterations\", \"progress\": 25, \"max_progress\": 100, \"current_error\": 3.8}\n\n",
                    "result_example": "data: {\"type\": \"result\", \"message\": \"Optimization completed successfully\", \"result\": {\"target_curve\": \"weighted_flat\", \"optimizer_preset\": \"default\", \"processing_time\": 18.7, \"final_rms_error\": 2.1, \"improvement_db\": 8.3, \"filters\": [...], \"frequency_response\": {...}}}\n\n"
                },
                "optimization_result": {
                    "description": "Complete EQ optimization results (delivered via streaming)",
                    "example": {
                        "target_curve": "weighted_flat",
                        "optimizer_preset": "default",
                        "processing_time": 18.7,
                        "final_rms_error": 2.1,
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
                            "frequencies": [20, 25, 31.5, 40, "..."],
                            "original_response": [-8.2, -7.1, -5.9, -4.8, "..."],
                            "corrected_response": [-0.8, -0.5, -0.2, 0.1, "..."],
                            "target_response": [0.0, 0.0, 0.0, 0.0, "..."]
                        },
                        "timestamp": "2025-08-18T14:30:15.123456"
                    }
                }
            }
        },
        "technical_details": {
            "audio_format": "16-bit PCM, 48kHz sample rate",
            "sweep_type": "Logarithmic frequency progression",
            "noise_type": "White noise with uniform frequency distribution and temporary file generation", 
            "microphone_calibration": "Automatic sensitivity and gain compensation",
            "device_detection": "ALSA-based audio device enumeration",
            "playback_monitoring": "Real-time status tracking with automatic timeout and file tracking",
            "fft_analysis": "FFT with windowing functions, normalization, and logarithmic frequency summarization",
            "fft_difference": "Signal comparison analysis with dB-domain difference computation, frequency interpolation, and statistical metrics",
            "frequency_summarization": "Logarithmic frequency buckets with configurable points per octave (1-100)",
            "difference_statistics": "RMS, maximum, mean difference calculations with frequency range analysis",
            "eq_optimization": "Rust-based high-performance optimization with multiple target curves, real-time progress reporting, and frequency response calculation",
            "eq_filters": "Biquad parametric EQ filters with peaking, high-pass, low-shelf, and high-shelf types",
            "optimization_algorithms": "Advanced curve fitting with frequency weighting and frequency-weighted error minimization",
            "file_management": "Automatic temporary file generation and cleanup for noise signals",
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
