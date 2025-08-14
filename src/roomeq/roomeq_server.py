#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
import logging

from .microphone import MicrophoneDetector, detect_microphones

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RoomEQ REST API",
    description="REST API for RoomEQ microphone detection and audio processing",
    version="0.1.0"
)

@app.get("/version")
def get_version():
    """Get API version information."""
    return {"version": "0.1.0"}


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
        for card_index, device_name, sensitivity in microphones:
            result.append({
                "card_index": card_index,
                "device_name": device_name,
                "sensitivity": float(sensitivity) if sensitivity != "0" else 0.0,
                "sensitivity_str": sensitivity
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


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "RoomEQ REST API",
        "version": "0.1.0",
        "endpoints": [
            "/version",
            "/microphones",
            "/microphones/raw", 
            "/audio/inputs",
            "/audio/cards",
            "/docs"
        ]
    }
