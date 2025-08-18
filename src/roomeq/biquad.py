"""
Simplified Biquad filter implementation for EQ optimization.

Copyright (c) 2025 HiFiBerry
"""

import math
import logging
import numpy as np
from typing import List, Optional, Tuple, Dict, Any


class Biquad:
    """Simplified biquad filter implementation for EQ optimization."""
    
    def __init__(self, a0: float, a1: float, a2: float, 
                 b0: float, b1: float, b2: float, 
                 description: str, filter_type: Optional[str] = None, 
                 f0: Optional[float] = None, q: Optional[float] = None, 
                 db: Optional[float] = None):
        """Initialize a biquad filter with coefficients and metadata."""
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.description = description
        self.filter_type = filter_type
        self.f0 = f0
        self.q = q
        self.db = db

    def coefficients_a(self, a0: bool = False) -> List[float]:
        """Get 'a' coefficients."""
        if a0:
            return [self.a0, self.a1, self.a2]
        else:
            return [self.a1, self.a2]

    def coefficients_b(self) -> List[float]:
        """Get 'b' coefficients."""
        return [self.b0, self.b1, self.b2]

    def to_dict(self) -> Dict[str, Any]:
        """Convert biquad to dictionary representation."""
        return {
            'filter_type': self.filter_type,
            'frequency': self.f0,
            'q': self.q,
            'gain_db': self.db,
            'description': self.description,
            'coefficients': {
                'b': self.coefficients_b(),
                'a': self.coefficients_a(a0=True)
            },
            'text_format': self.as_text()
        }

    def as_text(self) -> str:
        """Convert biquad to text representation."""
        if self.filter_type == "hp":
            return f"hp:{self.f0}:{self.q}"
        elif self.filter_type == "eq":
            return f"eq:{self.f0}:{self.q}:{self.db}"
        elif self.filter_type == "ls":
            return f"ls:{self.f0}:{self.q}:{self.db}"
        elif self.filter_type == "hs":
            return f"hs:{self.f0}:{self.q}:{self.db}"
        else:
            return f"coeff:{self.a0}:{self.a1}:{self.a2}:{self.b0}:{self.b1}:{self.b2}"

    @staticmethod
    def omega(f0: float, fs: float) -> float:
        """Calculate omega parameter."""
        return 2 * math.pi * f0 / fs

    @staticmethod
    def alpha(omega: float, q: float) -> float:
        """Calculate alpha parameter."""
        return math.sin(omega) / (2 * q)

    @staticmethod
    def a_factor(db_gain: float) -> float:
        """Calculate 'A' factor for gain."""
        return pow(10, db_gain / 40)

    @classmethod
    def high_pass(cls, f0: float, q: float, fs: float) -> 'Biquad':
        """Create high-pass filter."""
        w0 = cls.omega(f0, fs)
        alpha = cls.alpha(w0, q)
        b0 = (1 + math.cos(w0)) / 2
        b1 = -(1 + math.cos(w0))
        b2 = (1 + math.cos(w0)) / 2
        a0 = 1 + alpha
        a1 = -2 * math.cos(w0)
        a2 = 1 - alpha
        return cls(a0, a1, a2, b0, b1, b2, f"High pass {f0}Hz", "hp", f0, q)

    @classmethod
    def peaking_eq(cls, f0: float, q: float, db_gain: float, fs: float) -> 'Biquad':
        """Create peaking EQ filter."""
        w0 = cls.omega(f0, fs)
        alpha = cls.alpha(w0, q)
        a = cls.a_factor(db_gain)
        b0 = 1 + alpha * a
        b1 = -2 * math.cos(w0)
        b2 = 1 - alpha * a
        a0 = 1 + alpha / a
        a1 = -2 * math.cos(w0)
        a2 = 1 - alpha / a
        return cls(a0, a1, a2, b0, b1, b2, f"Peaking EQ {f0}Hz {db_gain}dB", "eq", f0, q, db_gain)

    @classmethod
    def low_shelf(cls, f0: float, q: float, db_gain: float, fs: float) -> 'Biquad':
        """Create low shelf filter."""
        w0 = cls.omega(f0, fs)
        alpha = cls.alpha(w0, q)
        a = cls.a_factor(db_gain)
        b0 = a * ((a + 1) - (a - 1) * math.cos(w0) + 2 * math.sqrt(a) * alpha)
        b1 = 2 * a * ((a - 1) - (a + 1) * math.cos(w0))
        b2 = a * ((a + 1) - (a - 1) * math.cos(w0) - 2 * math.sqrt(a) * alpha)
        a0 = (a + 1) + (a - 1) * math.cos(w0) + 2 * math.sqrt(a) * alpha
        a1 = -2 * ((a - 1) + (a + 1) * math.cos(w0))
        a2 = (a + 1) + (a - 1) * math.cos(w0) - 2 * math.sqrt(a) * alpha
        return cls(a0, a1, a2, b0, b1, b2, f"Low shelf {f0}Hz {db_gain}dB", "ls", f0, q, db_gain)

    @classmethod
    def high_shelf(cls, f0: float, q: float, db_gain: float, fs: float) -> 'Biquad':
        """Create high shelf filter."""
        w0 = cls.omega(f0, fs)
        alpha = cls.alpha(w0, q)
        a = cls.a_factor(db_gain)
        b0 = a * ((a + 1) + (a - 1) * math.cos(w0) + 2 * math.sqrt(a) * alpha)
        b1 = -2 * a * ((a - 1) + (a + 1) * math.cos(w0))
        b2 = a * ((a + 1) + (a - 1) * math.cos(w0) - 2 * math.sqrt(a) * alpha)
        a0 = (a + 1) - (a - 1) * math.cos(w0) + 2 * math.sqrt(a) * alpha
        a1 = 2 * ((a - 1) - (a + 1) * math.cos(w0))
        a2 = (a + 1) - (a - 1) * math.cos(w0) - 2 * math.sqrt(a) * alpha
        return cls(a0, a1, a2, b0, b1, b2, f"High shelf {f0}Hz {db_gain}dB", "hs", f0, q, db_gain)


def filter_cascade_response(frequencies: List[float], filters: List[Biquad], 
                          fs: float = 48000) -> Tuple[List[float], List[float]]:
    """
    Calculate magnitude and phase response of cascaded biquad filters.
    
    Args:
        frequencies: List of frequencies to evaluate
        filters: List of Biquad filters to cascade
        fs: Sample rate
        
    Returns:
        Tuple of (magnitude_db, phase_degrees)
    """
    from scipy.signal import freqz
    
    magnitudes = []
    phases = []
    
    for frequency in frequencies:
        total_mag_db = 0.0
        total_phase = 0.0
        
        for biquad in filters:
            # Calculate frequency response at this frequency
            w = 2 * math.pi * frequency / fs
            _, h = freqz([biquad.b0, biquad.b1, biquad.b2], 
                        [biquad.a0, biquad.a1, biquad.a2], 
                        worN=[w])
            
            # Accumulate magnitude (in dB) and phase
            mag_db = 20 * math.log10(abs(h[0]) + 1e-20)
            phase_rad = np.angle(h[0])
            
            total_mag_db += mag_db
            total_phase += phase_rad
            
        magnitudes.append(total_mag_db)
        phases.append(math.degrees(total_phase))
    
    return magnitudes, phases
