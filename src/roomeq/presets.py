"""
Centralized presets for RoomEQ optimizer - single source of truth.

This module contains all target curves and optimizer presets used throughout
the RoomEQ system, ensuring consistency between Rust and Python components.
"""

from typing import Dict, List, Union, Any


# Target curves for frequency response correction
TARGET_CURVES = {
    "flat": {
        "name": "Flat Response",
        "description": "Perfectly flat frequency response (0 dB across all frequencies)",
        "expert": False,
        "curve": [
            {"frequency": 20.0, "target_db": 0.0, "weight": None},
            {"frequency": 25000.0, "target_db": 0.0, "weight": None}
        ]
    },
    
    "weighted_flat": {
        "name": "Weighted Flat Response",
        "description": "Flat response with frequency weighting for optimal correction",
        "expert": True,
        "curve": [
            {"frequency": 20.0, "target_db": 0.0, "weight": [1.0, 0.3]},
            {"frequency": 100.0, "target_db": 0.0, "weight": [1.0, 0.6]},
            {"frequency": 200.0, "target_db": 0.0, "weight": [0.9, 0.7]},
            {"frequency": 500.0, "target_db": 0.0, "weight": 0.6},
            {"frequency": 5000.0, "target_db": 0.0, "weight": 0.4},
            {"frequency": 10000.0, "target_db": 0.0, "weight": 0.1},
            {"frequency": 25000.0, "target_db": 0.0, "weight": None}
        ]
    },
    
    "harman": {
        "name": "Harman Target Curve",
        "description": "Research-based preferred room response with gentle high-frequency roll-off",
        "expert": True,
        "curve": [
            {"frequency": 20.0, "target_db": 0.0, "weight": 0.8},
            {"frequency": 100.0, "target_db": 0.0, "weight": 1.0},
            {"frequency": 1000.0, "target_db": 0.0, "weight": 1.0},
            {"frequency": 10000.0, "target_db": -1.0, "weight": 0.5},
            {"frequency": 20000.0, "target_db": -2.0, "weight": 0.3}
        ]
    },
    
    "falling_slope": {
        "name": "Falling Slope",
        "description": "Gentle downward slope for warmer sound",
        "expert": False,
        "curve": [
            {"frequency": 20.0, "target_db": 2.0, "weight": 0.6},
            {"frequency": 100.0, "target_db": 1.0, "weight": 0.8},
            {"frequency": 1000.0, "target_db": 0.0, "weight": 1.0},
            {"frequency": 10000.0, "target_db": -2.0, "weight": 0.6},
            {"frequency": 20000.0, "target_db": -4.0, "weight": 0.3}
        ]
    },
    
    "room_only": {
        "name": "Room-Only Correction",
        "description": "Corrects only room-specific issues, preserves speaker character",
        "expert": True,
        "curve": [
            {"frequency": 20.0, "target_db": 0.0, "weight": 0.3},
            {"frequency": 200.0, "target_db": 0.0, "weight": 1.0},
            {"frequency": 2000.0, "target_db": 0.0, "weight": 1.0},
            {"frequency": 8000.0, "target_db": 0.0, "weight": 0.2},
            {"frequency": 20000.0, "target_db": 0.0, "weight": 0.1}
        ]
    },
    
    "vocal_presence": {
        "name": "Vocal Presence",
        "description": "Emphasizes vocal clarity and presence",
        "expert": False,
        "curve": [
            {"frequency": 20.0, "target_db": -1.0, "weight": 0.4},
            {"frequency": 200.0, "target_db": 0.0, "weight": 0.8},
            {"frequency": 1000.0, "target_db": 1.0, "weight": 1.2},
            {"frequency": 3000.0, "target_db": 1.5, "weight": 1.0},
            {"frequency": 8000.0, "target_db": 0.0, "weight": 0.6},
            {"frequency": 20000.0, "target_db": -1.0, "weight": 0.3}
        ]
    }
}


# Optimizer presets controlling correction aggressiveness and characteristics
OPTIMIZER_PRESETS = {
    "default": {
        "name": "Default",
        "description": "Balanced optimization with moderate Q values - recommended for most applications",
        "qmax": 10.0,
        "mindb": -10.0,
        "maxdb": 3.0,
        "add_highpass": True,
        "acceptable_error": 1.0
    },
    
    "verysmooth": {
        "name": "Very Smooth",
        "description": "Gentle corrections with very low Q values - minimal risk of artifacts",
        "qmax": 2.0,
        "mindb": -8.0,
        "maxdb": 2.0,
        "add_highpass": True,
        "acceptable_error": 2.0
    },
    
    "smooth": {
        "name": "Smooth",
        "description": "Moderate corrections with lower Q values - forgiving of measurement errors",
        "qmax": 5.0,
        "mindb": -8.0,
        "maxdb": 2.0,
        "add_highpass": True,
        "acceptable_error": 2.0
    },
    
    "aggressive": {
        "name": "Aggressive",
        "description": "Strong corrections with high Q values - requires accurate measurements",
        "qmax": 15.0,
        "mindb": -15.0,
        "maxdb": 5.0,
        "add_highpass": True,
        "acceptable_error": 0
    },
    
    "precise": {
        "name": "Precise",
        "description": "Maximum precision with highest Q values - for expert use with excellent measurements",
        "qmax": 20.0,
        "mindb": -20.0,
        "maxdb": 6.0,
        "add_highpass": True,
        "acceptable_error": 0
    },
    
    "no_highpass": {
        "name": "No High-Pass",
        "description": "Default settings without automatic high-pass filter",
        "qmax": 10.0,
        "mindb": -10.0,
        "maxdb": 3.0,
        "add_highpass": False,
        "acceptable_error": 2.0
    }
}


def list_target_curves():
    """Get list of available target curves with metadata."""
    return [{
        "key": name,
        "name": curve["name"],
        "description": curve["description"],
        "expert": curve.get("expert", False),
        "curve": curve["curve"]
    } for name, curve in TARGET_CURVES.items()]


def list_optimizer_presets():
    """Get list of available optimizer presets with metadata."""
    return [{
        "key": name,
        "preset": name,
        "name": preset["name"],
        "description": preset["description"],
        "qmax": preset["qmax"],
        "mindb": preset["mindb"],
        "maxdb": preset["maxdb"],
        "add_highpass": preset["add_highpass"],
        "acceptable_error": preset["acceptable_error"]
    } for name, preset in OPTIMIZER_PRESETS.items()]


def get_target_curve(name: str) -> Dict[str, Any]:
    """Get target curve data by name."""
    if name not in TARGET_CURVES:
        raise ValueError(f"Unknown target curve: {name}. Available: {list(TARGET_CURVES.keys())}")
    return TARGET_CURVES[name].copy()


def get_optimizer_preset(name: str) -> Dict[str, Any]:
    """Get optimizer preset by name.""" 
    if name not in OPTIMIZER_PRESETS:
        raise ValueError(f"Unknown optimizer preset: {name}. Available: {list(OPTIMIZER_PRESETS.keys())}")
    return OPTIMIZER_PRESETS[name].copy()


def validate_target_curve_name(name: str) -> bool:
    """Check if target curve name is valid."""
    return name in TARGET_CURVES


def validate_optimizer_preset_name(name: str) -> bool:
    """Check if optimizer preset name is valid."""
    return name in OPTIMIZER_PRESETS
