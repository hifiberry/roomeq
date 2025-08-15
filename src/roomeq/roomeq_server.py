#!/usr/bin/env python3
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from typing import List, Dict, Any, Optional
import logging
import threading
import time
from datetime import datetime, timedelta

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
    'device': None
}

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Add response headers middleware
@app.after_request
def after_request(response):
    """Add headers to all responses for better proxy compatibility."""
    response.headers["connection"] = "close"
    response.headers["server"] = "roomeq-api/0.2.0"
    return response


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
        "version": "0.2.0",
        "api_name": "RoomEQ Audio Processing API",
        "features": [
            "Microphone detection with sensitivity and gain",
            "SPL measurement",
            "White noise generation with keep-alive control",
            "Real-time playback management"
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
            'device': device
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
    """Get the current status of noise playbook."""
    global _playback_state
    
    if _playback_state['active'] and _playback_state['stop_time']:
        remaining_time = (_playback_state['stop_time'] - datetime.now()).total_seconds()
        remaining_time = max(0, remaining_time)  # Don't show negative time
    else:
        remaining_time = 0
    
    return jsonify({
        "active": _playback_state['active'],
        "amplitude": _playback_state['amplitude'],
        "device": _playback_state['device'] or "default",
        "remaining_seconds": round(remaining_time, 1),
        "stop_time": _playback_state['stop_time'].isoformat() if _playback_state['stop_time'] else None
    })


@app.route("/", methods=["GET"])
def root():
    """Root endpoint with API information."""
    return jsonify({
        "message": "RoomEQ Audio Processing API",
        "version": "0.2.0",
        "description": "REST API for microphone detection, SPL measurement, and audio signal generation",
        "endpoints": {
            "info": ["/", "/version"],
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
    })


def main():
    """Main entry point for the roomeq-server console script."""
    app.run(
        host="0.0.0.0", 
        port=10315,
        debug=False,
        threaded=True
    )
