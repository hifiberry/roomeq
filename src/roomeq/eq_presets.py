"""
EQ optimization target curves and optimizer presets.

Copyright (c) 2025 HiFiBerry

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Dict, List, Union, Tuple, Any

# Optimizer configurations
OPTIMIZER_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {
        "name": "Default",
        "description": "Balanced optimization with moderate Q values",
        "qmax": 10,
        "mindb": -10,
        "maxdb": 3,
        "add_highpass": True,
    },
    "verysmooth": {
        "name": "Very Smooth",
        "description": "Gentle correction with low Q values",
        "qmax": 2,
        "mindb": -10,
        "maxdb": 3,
        "add_highpass": True,
    },
    "smooth": {
        "name": "Smooth",
        "description": "Moderate correction with medium Q values",
        "qmax": 5,
        "mindb": -10,
        "maxdb": 3,
        "add_highpass": True,
    },
    "aggressive": {
        "name": "Aggressive",
        "description": "Strong correction with high Q values",
        "qmax": 20,
        "mindb": -20,
        "maxdb": 6,
        "add_highpass": True,
    },
    "precise": {
        "name": "Precise",
        "description": "High precision correction for detailed tuning",
        "qmax": 15,
        "mindb": -15,
        "maxdb": 5,
        "add_highpass": False,
    },
}

# Target curve definitions
# Format: [[frequency, target_db, (positive_weight, negative_weight, cutoff)], ...]
# Weight can be a single number or tuple (pos_weight, neg_weight) or (pos_weight, neg_weight, cutoff)
TARGET_CURVES: Dict[str, List[List[Union[float, Tuple]]]] = {
    "flat": {
        "name": "Flat Response",
        "description": "Flat frequency response across all frequencies",
        "curve": [
            [20, 0],
            [25000, 0]
        ]
    },
    "weighted_flat": {
        "name": "Weighted Flat",
        "description": "Flat response with frequency-dependent correction weights",
        "curve": [
            [20, 0, (1, 0.3)],
            [100, 0, (1, 0.6)],
            [200, 0, (0.9, 0.7)],
            [500, 0, 0.6],
            [5000, 0, 0.4],
            [10000, 0, 0.1],
            [25000, 0]
        ]
    },
    "falling_slope": {
        "name": "Falling Slope",
        "description": "Gentle high-frequency roll-off for warmer sound",
        "curve": [
            [20, 0, (1, 0.3)],
            [100, 0, (1, 0.6)],
            [200, 0, (0.9, 0.7)],
            [500, 0, (0.9, 0.5)],
            [1000, 0, (0.5, 0.3)],
            [10000, -3, 0.1],
            [25000, -6]
        ]
    },
    "room_only": {
        "name": "Room Correction Only",
        "description": "Focus correction on room modes (low frequencies)",
        "curve": [
            [20, 0, (1, 0.5)],
            [250, 0]
        ]
    },
    "harman": {
        "name": "Harman Curve",
        "description": "Harman research target curve for speakers",
        "curve": [
            [20, -2, (0.8, 0.4)],
            [100, 0, (1, 0.7)],
            [200, 0, (1, 0.8)],
            [500, 0, (0.9, 0.7)],
            [1000, 0, (0.7, 0.5)],
            [3000, -1, (0.6, 0.4)],
            [10000, -3, (0.3, 0.2)],
            [20000, -5, (0.1, 0.1)]
        ]
    },
    "vocal_presence": {
        "name": "Vocal Presence",
        "description": "Enhanced midrange for vocal clarity",
        "curve": [
            [20, -2, (0.5, 0.3)],
            [100, -1, (0.7, 0.5)],
            [200, 0, (0.9, 0.7)],
            [500, 0, (1, 0.8)],
            [1000, 1, (1, 0.9)],
            [3000, 2, (1, 0.8)],
            [5000, 0, (0.8, 0.6)],
            [10000, -2, (0.4, 0.3)],
            [20000, -4, (0.2, 0.1)]
        ]
    }
}

# Recommended target curves in order of preference
RECOMMENDED_TARGETS = [
    "room_only",
    "weighted_flat", 
    "falling_slope",
    "harman",
    "flat",
    "vocal_presence"
]

def get_target_curve_info(curve_name: str) -> Dict[str, Any]:
    """Get information about a target curve."""
    if curve_name not in TARGET_CURVES:
        raise ValueError(f"Unknown target curve: {curve_name}")
    
    return {
        "name": curve_name,
        **TARGET_CURVES[curve_name]
    }

def get_optimizer_info(preset_name: str) -> Dict[str, Any]:
    """Get information about an optimizer preset."""
    if preset_name not in OPTIMIZER_PRESETS:
        raise ValueError(f"Unknown optimizer preset: {preset_name}")
        
    return {
        "preset": preset_name,
        **OPTIMIZER_PRESETS[preset_name]
    }

def list_target_curves() -> List[Dict[str, Any]]:
    """List all available target curves."""
    return [get_target_curve_info(name) for name in TARGET_CURVES.keys()]

def list_optimizer_presets() -> List[Dict[str, Any]]:
    """List all available optimizer presets.""" 
    return [get_optimizer_info(name) for name in OPTIMIZER_PRESETS.keys()]
