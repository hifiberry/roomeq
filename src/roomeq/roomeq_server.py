#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException, Query
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

app = FastAPI(
    title="RoomEQ Audio Processing API",
    description="""
    REST API for RoomEQ microphone detection, SPL measurement, and audio signal generation.
    
    This API provides endpoints for:
    - Detecting available microphones with sensitivity and gain information
    - Measuring sound pressure levels (SPL) using detected microphones  
    - Generating test signals (noise, sine sweeps) for acoustic measurements
    - Real-time signal playback control with keep-alive functionality
    
    Designed for acoustic measurement systems, room correction, and audio testing applications.
    """,
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/version")
def get_version():
    """Get API version information."""
    return {
        "version": "0.2.0",
        "api_name": "RoomEQ Audio Processing API",
        "features": [
            "Microphone detection with sensitivity and gain",
            "SPL measurement",
            "White noise generation with keep-alive control",
            "Real-time playback management"
        ]
    }


@app.get("/microphones", response_model=List[Dict[str, Any]])
def get_microphones():
    """
    Get detected microphones with their properties.
    
    Returns a list of microphone objects with card index, device name, and sensitivity.
    """
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
        
        return result
    
    except Exception as e:
        logger.error(f"Error detecting microphones: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to detect microphones: {str(e)}")


@app.get("/microphones/raw")
def get_microphones_raw():
    """
    Get detected microphones in raw format (compatible with bash script output).
    
    Returns a list of strings in format: "card_index:device_name:sensitivity"
    """
    try:
        microphones = detect_microphones()
        return {"microphones": microphones}
    
    except Exception as e:
        logger.error(f"Error detecting microphones: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to detect microphones: {str(e)}")


@app.get("/audio/inputs")
def get_audio_inputs():
    """
    Get audio input card indices.
    
    Returns a list of card indices that have input (capture) capability.
    """
    try:
        detector = MicrophoneDetector()
        input_cards = detector.get_audio_inputs()
        
        return {
            "input_cards": input_cards,
            "count": len(input_cards)
        }
    
    except Exception as e:
        logger.error(f"Error getting audio inputs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get audio inputs: {str(e)}")


@app.get("/audio/cards")
def get_audio_cards():
    """
    Get all available audio cards.
    
    Returns a list of all audio cards detected by ALSA.
    """
    try:
        detector = MicrophoneDetector()
        cards = detector.audio_cards
        
        return {
            "cards": cards,
            "count": len(cards)
        }
    
    except Exception as e:
        logger.error(f"Error getting audio cards: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get audio cards: {str(e)}")


@app.get("/spl/measure")
def measure_spl_level(
    device: Optional[str] = Query(None, description="ALSA device name (e.g., hw:1,0). If not specified, auto-detects first microphone"),
    duration: float = Query(1.0, description="Measurement duration in seconds", ge=0.1, le=10.0)
):
    """
    Measure SPL level using the specified or auto-detected microphone.
    
    Returns SPL measurement with detailed information about the measurement.
    """
    try:
        result = measure_spl(device=device, duration=duration)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Calculate effective sensitivity for display
        effective_sensitivity = None
        if result['microphone_sensitivity'] and result['microphone_gain'] is not None:
            effective_sensitivity = result['microphone_sensitivity'] - result['microphone_gain']
        
        return {
            "spl_db": result['spl_db'],
            "rms_db_fs": result['rms_db_fs'],
            "device": result['device'],
            "duration": result['duration'],
            "microphone": {
                "sensitivity": result['microphone_sensitivity'],
                "gain_db": result['microphone_gain'],
                "effective_sensitivity": effective_sensitivity
            },
            "timestamp": __import__('time').time(),
            "success": result['success']
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error measuring SPL: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to measure SPL: {str(e)}")


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


@app.post("/audio/noise/start")
def start_noise(
    duration: float = Query(3.0, description="Duration in seconds to play noise", ge=1.0, le=30.0),
    amplitude: float = Query(0.5, description="Amplitude (0.0-1.0)", ge=0.0, le=1.0),
    device: Optional[str] = Query(None, description="Output device (e.g., hw:0,0). If not specified, uses default")
):
    """
    Start playing white noise for the specified duration.
    
    This endpoint starts noise playback that will automatically stop after the specified duration
    unless extended by calling the keep-playing endpoint.
    """
    global _signal_generator, _playback_state
    
    try:
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
        
        return {
            "status": "started",
            "duration": duration,
            "amplitude": amplitude,
            "device": device or "default",
            "stop_time": stop_time.isoformat(),
            "message": f"Noise playback started for {duration} seconds"
        }
        
    except Exception as e:
        logger.error(f"Error starting noise: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start noise: {str(e)}")


@app.post("/audio/noise/keep-playing")
def keep_playing_noise(
    duration: float = Query(3.0, description="Additional duration in seconds", ge=1.0, le=30.0)
):
    """
    Extend the current noise playback by the specified duration.
    
    This endpoint extends the current playback time, allowing for continuous playback
    as long as keep-alive requests are sent regularly. The playback will stop
    automatically if no keep-alive request is received before the current stop time.
    """
    global _playback_state
    
    if not _playback_state['active']:
        raise HTTPException(status_code=404, detail="No active noise playback to extend")
    
    try:
        # Extend the stop time
        new_stop_time = datetime.now() + timedelta(seconds=duration)
        _playback_state['stop_time'] = new_stop_time
        
        logger.info(f"Extended noise playback by {duration} seconds")
        
        return {
            "status": "extended",
            "duration": duration,
            "new_stop_time": new_stop_time.isoformat(),
            "message": f"Playback extended by {duration} seconds"
        }
        
    except Exception as e:
        logger.error(f"Error extending playback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extend playback: {str(e)}")


@app.post("/audio/noise/stop")
def stop_noise():
    """
    Stop the current noise playback immediately.
    """
    global _signal_generator, _playback_state
    
    try:
        if _signal_generator and _playback_state['active']:
            _signal_generator.stop()
            _playback_state['active'] = False
            _playback_state['stop_time'] = None
            
            logger.info("Stopped noise playback")
            
            return {
                "status": "stopped",
                "message": "Noise playback stopped"
            }
        else:
            return {
                "status": "not_active",
                "message": "No active noise playback to stop"
            }
            
    except Exception as e:
        logger.error(f"Error stopping noise: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop noise: {str(e)}")


@app.get("/audio/noise/status")
def get_noise_status():
    """
    Get the current status of noise playback.
    """
    global _playback_state
    
    if _playback_state['active'] and _playback_state['stop_time']:
        remaining_time = (_playback_state['stop_time'] - datetime.now()).total_seconds()
        remaining_time = max(0, remaining_time)  # Don't show negative time
    else:
        remaining_time = 0
    
    return {
        "active": _playback_state['active'],
        "amplitude": _playback_state['amplitude'],
        "device": _playback_state['device'] or "default",
        "remaining_seconds": round(remaining_time, 1),
        "stop_time": _playback_state['stop_time'].isoformat() if _playback_state['stop_time'] else None
    }


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10315)
