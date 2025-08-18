"""
Modern room EQ optimizer with real-time progress reporting.

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

import numpy as np
import logging
import math
import threading
import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Tuple
from scipy.signal import freqz
from scipy import optimize
from dataclasses import dataclass, asdict

from .biquad import Biquad, filter_cascade_response
from .eq_presets import OPTIMIZER_PRESETS, TARGET_CURVES

logger = logging.getLogger(__name__)


@dataclass
class OptimizationStep:
    """Represents a single step in the optimization process."""
    step_number: int
    step_type: str  # 'initialization', 'filter_optimization', 'completion'
    message: str
    progress_percent: float
    current_error: Optional[float] = None
    best_filter: Optional[Dict[str, Any]] = None
    total_filters: Optional[int] = None
    filter_count: Optional[int] = None
    frequency_range: Optional[Tuple[float, float]] = None
    target_curve: Optional[str] = None


@dataclass
class OptimizationResult:
    """Complete optimization result."""
    optimization_id: str
    success: bool
    filters: List[Dict[str, Any]]
    final_error: float
    original_error: float
    improvement_db: float
    frequency_range: Tuple[float, float]
    target_curve: str
    optimizer_preset: str
    processing_time: float
    steps: List[OptimizationStep]
    frequencies: List[float]
    original_response: List[float]
    target_response: List[float]
    corrected_response: List[float]
    intermediate_results: List['OptimizationResult'] = None
    error_message: Optional[str] = None


class ProgressReporter:
    """Handles progress reporting for optimization."""
    
    def __init__(self, callback: Optional[Callable[[OptimizationStep], None]] = None):
        self.callback = callback
        self.steps: List[OptimizationStep] = []
    
    def report_step(self, step: OptimizationStep):
        """Report a new optimization step."""
        self.steps.append(step)
        logger.info(f"Optimization step {step.step_number}: {step.message}")
        if self.callback:
            try:
                self.callback(step)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")


def mag2db(magnitude: float) -> float:
    """Convert magnitude to dB."""
    return 20 * np.log10(np.maximum(magnitude, 1e-20))


def filter_cascade_response(frequencies: List[float], filters: List[Biquad], 
                          fs: float = 48000) -> Tuple[List[float], List[float]]:
    """
    Calculate magnitude and phase response of cascaded biquad filters.
    """
    from scipy.signal import freqz
    
    freq_array = np.array(frequencies)
    total_magnitude_db = np.zeros_like(freq_array)
    total_phase_degrees = np.zeros_like(freq_array)
    
    for frequency_idx, frequency in enumerate(frequencies):
        for biquad in filters:
            # Calculate frequency response at this frequency
            w = 2 * np.pi * frequency / fs
            _, h = freqz([biquad.b0, biquad.b1, biquad.b2], 
                        [biquad.a0, biquad.a1, biquad.a2], 
                        worN=[w])
            
            # Accumulate magnitude (in dB) and phase
            mag_db = 20 * np.log10(np.abs(h[0]) + 1e-20)
            phase_rad = np.angle(h[0])
            
            total_magnitude_db[frequency_idx] += mag_db
            total_phase_degrees[frequency_idx] += np.degrees(phase_rad)
    
    return total_magnitude_db.tolist(), total_phase_degrees.tolist()


def db2mag(db: float) -> float:
    """Convert decibels to magnitude."""
    return 10 ** (db / 20)


def calculate_rms_error(curve1: List[float], curve2: List[float], 
                       weights: Optional[List[float]] = None) -> float:
    """Calculate RMS error between two curves with optional weighting."""
    if len(curve1) != len(curve2):
        raise ValueError("Curves must have the same length")
    
    rms = 0.0
    total_weight = 0.0
    
    for i, (c1, c2) in enumerate(zip(curve1, curve2)):
        diff = c1 - c2
        weight = weights[i] if weights else 1.0
        if isinstance(weight, (list, tuple)):
            weight = weight[0] if diff >= 0 else weight[1]
        
        rms += (diff ** 2) * weight
        total_weight += weight
    
    return math.sqrt(rms / total_weight) if total_weight > 0 else 0.0


def normalize_response(frequencies: np.ndarray, magnitudes: np.ndarray, 
                      normalize_freq: float = 1000.0) -> np.ndarray:
    """Normalize response to a reference frequency."""
    # Find closest frequency index
    ref_idx = np.argmin(np.abs(frequencies - normalize_freq))
    ref_level = magnitudes[ref_idx]
    return magnitudes - ref_level


def find_usable_range(frequencies: np.ndarray, magnitudes: np.ndarray,
                     min_db: float = -8, avg_range: Tuple[float, float] = (200, 8000),
                     max_fails: int = 2) -> Tuple[int, int, float, float]:
    """Find the usable frequency range for optimization."""
    # Calculate average level in the specified range
    avg_mask = (frequencies >= avg_range[0]) & (frequencies <= avg_range[1])
    if not np.any(avg_mask):
        raise ValueError(f"No frequencies found in average range {avg_range}")
    
    avg_db = np.mean(magnitudes[avg_mask])
    normalized_mag = magnitudes - avg_db
    
    # Find center frequency index (1kHz)
    center_idx = np.argmin(np.abs(frequencies - 1000.0))
    
    # Find usable range going down from center
    fails = 0
    low_idx = 0
    for i in range(center_idx, 0, -1):
        if normalized_mag[i] < min_db:
            fails += 1
        else:
            fails = 0
        if fails >= max_fails:
            low_idx = i + max_fails
            break
    
    # Find usable range going up from center  
    fails = 0
    high_idx = len(frequencies) - 1
    for i in range(center_idx, len(frequencies)):
        if normalized_mag[i] < min_db:
            fails += 1
        else:
            fails = 0
        if fails >= max_fails:
            high_idx = i - max_fails
            break
    
    return low_idx, high_idx, frequencies[low_idx], frequencies[high_idx]


def generate_target_curve(frequencies: np.ndarray, 
                         target_definition: List[List]) -> Tuple[List[float], List[float]]:
    """Generate target curve and weights from definition."""
    target_db = []
    weights = []
    
    for frequency in frequencies:
        # Find the target segment for this frequency
        target_db_val = 0.0
        weight_val = 1.0
        
        # Find the two target points that bracket this frequency
        for i in range(len(target_definition) - 1):
            f1, db1 = target_definition[i][:2]
            f2, db2 = target_definition[i + 1][:2]
            
            # Get weight from first point
            if len(target_definition[i]) > 2:
                weight_spec = target_definition[i][2]
                if isinstance(weight_spec, (list, tuple)):
                    weight_val = weight_spec
                else:
                    weight_val = weight_spec
            
            if f1 <= frequency <= f2:
                # Linear interpolation in log frequency domain
                log_freq = math.log10(frequency)
                log_f1 = math.log10(f1)
                log_f2 = math.log10(f2)
                
                if log_f2 != log_f1:
                    interpolation_factor = (log_freq - log_f1) / (log_f2 - log_f1)
                    target_db_val = db1 + (db2 - db1) * interpolation_factor
                else:
                    target_db_val = db1
                break
        
        target_db.append(target_db_val)
        weights.append(weight_val)
    
    return target_db, weights


def optimize_single_filter(frequency: float, frequencies: np.ndarray,
                          input_curve: np.ndarray, target_curve: np.ndarray,
                          target_weights: List, optimizer_params: Dict[str, Any],
                          fs: float = 48000) -> Tuple[float, float, float, float]:
    """Optimize a single peaking EQ filter at a specific frequency."""
    
    def error_function(params):
        q, gain = params
        biquad = Biquad.peaking_eq(frequency, q, gain, fs)
        
        # Calculate filter response
        response_change = []
        for freq in frequencies:
            w = 2 * math.pi * freq / fs
            _, h = freqz([biquad.b0, biquad.b1, biquad.b2],
                        [biquad.a0, biquad.a1, biquad.a2], worN=[w])
            response_change.append(mag2db(abs(h[0])))
        
        # Apply filter to input curve
        corrected_curve = input_curve + np.array(response_change)
        
        # Calculate weighted error
        errors = []
        for i, (corrected, target) in enumerate(zip(corrected_curve, target_curve)):
            diff = corrected - target
            weight = target_weights[i]
            if isinstance(weight, (list, tuple)):
                w = weight[0] if diff >= 0 else weight[1]
            else:
                w = weight
            errors.append(diff * w)
        
        return errors
    
    # Optimization bounds
    q_bounds = (0.1, optimizer_params.get('qmax', 10))
    gain_bounds = (optimizer_params.get('mindb', -15), optimizer_params.get('maxdb', 5))
    
    result = optimize.least_squares(
        error_function,
        [1.0, 0.0],  # Initial guess: Q=1, gain=0dB
        bounds=([q_bounds[0], gain_bounds[0]], [q_bounds[1], gain_bounds[1]]),
        max_nfev=1000
    )
    
    q_optimal, gain_optimal = result.x
    final_error = calculate_rms_error(result.fun, [0] * len(result.fun))
    
    return frequency, q_optimal, gain_optimal, final_error


class RoomEQOptimizer:
    """Modern room EQ optimizer with progress reporting."""
    
    def __init__(self, progress_callback: Optional[Callable[[OptimizationStep], None]] = None):
        self.progress_reporter = ProgressReporter(progress_callback)
        self.optimization_id = str(uuid.uuid4())
        self.is_cancelled = False
    
    def cancel(self):
        """Cancel the current optimization."""
        self.is_cancelled = True
    
    def optimize(self, frequencies: List[float], magnitudes: List[float],
                target_curve: str = "weighted_flat", optimizer_preset: str = "default",
                filter_count: int = 8, sample_rate: float = 48000, 
                intermediate_results_interval: int = 0) -> OptimizationResult:
        """
        Optimize EQ filters for the given frequency response.
        
        Args:
            frequencies: Frequency array from measurement
            magnitudes: Magnitude response in dB
            target_curve: Target curve name from eq_presets
            optimizer_preset: Optimizer preset name from eq_presets
            filter_count: Number of EQ filters to generate
            sample_rate: Audio sample rate
            intermediate_results_interval: Return intermediate results every n steps (0=disabled)
            
        Returns:
            OptimizationResult with complete optimization data
        """
        start_time = time.time()
        
        try:
            # Step 1: Initialization
            self.progress_reporter.report_step(OptimizationStep(
                step_number=1,
                step_type="initialization",
                message="Initializing optimization parameters",
                progress_percent=0.0,
                target_curve=target_curve
            ))
            
            if self.is_cancelled:
                raise RuntimeError("Optimization cancelled")
            
            # Convert to numpy arrays
            freq_array = np.array(frequencies)
            mag_array = np.array(magnitudes)
            
            # Get configuration
            if target_curve not in TARGET_CURVES:
                raise ValueError(f"Unknown target curve: {target_curve}")
            if optimizer_preset not in OPTIMIZER_PRESETS:
                raise ValueError(f"Unknown optimizer preset: {optimizer_preset}")
            
            target_config = TARGET_CURVES[target_curve]
            optimizer_params = OPTIMIZER_PRESETS[optimizer_preset]
            
            # Step 2: Normalize and find usable range
            self.progress_reporter.report_step(OptimizationStep(
                step_number=2,
                step_type="initialization", 
                message="Normalizing response and finding usable frequency range",
                progress_percent=10.0
            ))
            
            # Normalize to average level
            avg_mask = (freq_array >= 200) & (freq_array <= 8000)
            if not np.any(avg_mask):
                raise ValueError("No frequencies found in normalization range")
            
            avg_level = np.mean(mag_array[avg_mask])
            normalized_mag = mag_array - avg_level
            
            # Find usable range
            max_fails = 2 if len(frequencies) >= 64 else 1
            low_idx, high_idx, f_low, f_high = find_usable_range(
                freq_array, normalized_mag, max_fails=max_fails
            )
            
            # Step 3: Generate target curve
            self.progress_reporter.report_step(OptimizationStep(
                step_number=3,
                step_type="initialization",
                message=f"Generating target curve: {target_config['name']}",
                progress_percent=15.0,
                frequency_range=(f_low, f_high)
            ))
            
            # Extract frequency range for optimization
            opt_frequencies = freq_array[low_idx:high_idx]
            opt_magnitudes = normalized_mag[low_idx:high_idx]
            
            # Generate target curve for this frequency range
            target_response, target_weights = generate_target_curve(
                opt_frequencies, target_config['curve']
            )
            
            # Calculate initial error
            initial_error = calculate_rms_error(opt_magnitudes, target_response, target_weights)
            
            self.progress_reporter.report_step(OptimizationStep(
                step_number=4,
                step_type="initialization",
                message=f"Initial RMS error: {initial_error:.3f} dB",
                progress_percent=20.0,
                current_error=initial_error
            ))
            
            # Step 4: Add high-pass filter if requested
            filters = []
            current_response = opt_magnitudes.copy()
            
            if optimizer_params.get('add_highpass', False):
                self.progress_reporter.report_step(OptimizationStep(
                    step_number=5,
                    step_type="filter_optimization",
                    message="Adding high-pass filter for low frequency control",
                    progress_percent=25.0
                ))
                
                hp_freq = f_low / 2
                hp_filter = Biquad.high_pass(hp_freq, 0.5, sample_rate)
                filters.append(hp_filter)
                
                # Apply high-pass response
                hp_response, _ = filter_cascade_response(opt_frequencies.tolist(), [hp_filter], sample_rate)
                current_response = current_response + np.array(hp_response)
            
            # Step 5: Optimize peaking EQ filters
            intermediate_results = []
            for filter_num in range(filter_count):
                if self.is_cancelled:
                    raise RuntimeError("Optimization cancelled")
                
                step_num = 6 + filter_num
                progress = 25.0 + (filter_num / filter_count) * 60.0
                
                self.progress_reporter.report_step(OptimizationStep(
                    step_number=step_num,
                    step_type="filter_optimization",
                    message=f"Optimizing EQ filter {filter_num + 1}/{filter_count}",
                    progress_percent=progress,
                    filter_count=filter_num,
                    total_filters=filter_count
                ))
                
                # Find optimal filter across all frequencies
                best_error = float('inf')
                best_filter = None
                best_freq = None
                
                # Test optimization at multiple frequency points (subsample for performance)
                test_frequencies = opt_frequencies[::max(1, len(opt_frequencies) // 20)]
                
                for test_freq in test_frequencies:
                    if self.is_cancelled:
                        break
                        
                    freq, q, gain, error = optimize_single_filter(
                        test_freq, opt_frequencies, current_response,
                        target_response, target_weights, optimizer_params, sample_rate
                    )
                    
                    if error < best_error:
                        best_error = error
                        best_freq = freq
                        best_filter = Biquad.peaking_eq(freq, q, gain, sample_rate)
                
                if best_filter is None:
                    logger.warning(f"Could not optimize filter {filter_num + 1}")
                    continue
                
                # Apply the best filter
                filters.append(best_filter)
                filter_response, _ = filter_cascade_response(opt_frequencies.tolist(), [best_filter], sample_rate)
                current_response = current_response + np.array(filter_response)
                
                # Calculate new error
                new_error = calculate_rms_error(current_response, target_response, target_weights)
                
                self.progress_reporter.report_step(OptimizationStep(
                    step_number=step_num + 100,  # Sub-step
                    step_type="filter_optimization",
                    message=f"Filter {filter_num + 1}: {best_freq:.1f}Hz, Q={best_filter.q:.2f}, {best_filter.db:.2f}dB (Error: {new_error:.3f})",
                    progress_percent=progress + (60.0 / filter_count * 0.5),
                    current_error=new_error,
                    best_filter=best_filter.to_dict()
                ))
                
                # Check if we should return intermediate results
                if (intermediate_results_interval > 0 and 
                    (filter_num + 1) % intermediate_results_interval == 0 and 
                    filter_num + 1 < filter_count):
                    
                    # Calculate current total response with all filters applied so far
                    current_total_response, _ = filter_cascade_response(opt_frequencies.tolist(), filters, sample_rate)
                    current_corrected = opt_magnitudes + np.array(current_total_response)
                    current_error = calculate_rms_error(current_corrected, target_response, target_weights)
                    current_improvement = initial_error - current_error
                    
                    # Create intermediate result
                    intermediate_result = OptimizationResult(
                        optimization_id=self.optimization_id + f"_step_{filter_num + 1}",
                        success=True,
                        filters=[f.to_dict() for f in filters],
                        final_error=current_error,
                        original_error=initial_error,
                        improvement_db=current_improvement,
                        frequency_range=(f_low, f_high),
                        target_curve=target_curve,
                        optimizer_preset=optimizer_preset,
                        processing_time=time.time() - start_time,
                        steps=self.progress_reporter.steps.copy(),
                        frequencies=opt_frequencies.tolist(),
                        original_response=opt_magnitudes.tolist(),
                        target_response=target_response,
                        corrected_response=current_corrected.tolist()
                    )
                    
                    # Store intermediate result
                    intermediate_results.append(intermediate_result)
                    
                    # Report intermediate result via callback
                    self.progress_reporter.report_step(OptimizationStep(
                        step_number=step_num + 200,  # Intermediate result step
                        step_type="intermediate_result",
                        message=f"Intermediate result after {filter_num + 1} filters: {current_improvement:.2f} dB improvement",
                        progress_percent=progress + (60.0 / filter_count * 0.8),
                        current_error=current_error
                    ))
            
            # Step 6: Generate final results
            if self.is_cancelled:
                raise RuntimeError("Optimization cancelled")
            
            self.progress_reporter.report_step(OptimizationStep(
                step_number=200,
                step_type="completion",
                message="Generating final results and responses",
                progress_percent=90.0
            ))
            
            # Calculate final response with all filters
            final_response, _ = filter_cascade_response(opt_frequencies.tolist(), filters, sample_rate)
            final_corrected = opt_magnitudes + np.array(final_response)
            final_error = calculate_rms_error(final_corrected, target_response, target_weights)
            
            processing_time = time.time() - start_time
            improvement_db = initial_error - final_error
            
            # Final step
            self.progress_reporter.report_step(OptimizationStep(
                step_number=201,
                step_type="completion",
                message=f"Optimization completed! Improvement: {improvement_db:.2f} dB ({processing_time:.1f}s)",
                progress_percent=100.0,
                current_error=final_error
            ))
            
            # Create result
            result = OptimizationResult(
                optimization_id=self.optimization_id,
                success=True,
                filters=[f.to_dict() for f in filters],
                final_error=final_error,
                original_error=initial_error,
                improvement_db=improvement_db,
                frequency_range=(f_low, f_high),
                target_curve=target_curve,
                optimizer_preset=optimizer_preset,
                processing_time=processing_time,
                steps=self.progress_reporter.steps,
                frequencies=opt_frequencies.tolist(),
                original_response=opt_magnitudes.tolist(),
                target_response=target_response,
                corrected_response=final_corrected.tolist(),
                intermediate_results=intermediate_results if intermediate_results else None
            )
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Optimization failed: {error_msg}")
            
            self.progress_reporter.report_step(OptimizationStep(
                step_number=999,
                step_type="completion", 
                message=f"Optimization failed: {error_msg}",
                progress_percent=0.0
            ))
            
            processing_time = time.time() - start_time
            
            return OptimizationResult(
                optimization_id=self.optimization_id,
                success=False,
                filters=[],
                final_error=0.0,
                original_error=0.0,
                improvement_db=0.0,
                frequency_range=(0.0, 0.0),
                target_curve=target_curve,
                optimizer_preset=optimizer_preset,
                processing_time=processing_time,
                steps=self.progress_reporter.steps,
                frequencies=[],
                original_response=[],
                target_response=[],
                corrected_response=[],
                intermediate_results=None,
                error_message=error_msg
            )


# Global state for tracking active optimizations
_active_optimizations: Dict[str, RoomEQOptimizer] = {}
_optimization_results: Dict[str, OptimizationResult] = {}


def start_optimization(frequencies: List[float], magnitudes: List[float],
                      target_curve: str = "weighted_flat", optimizer_preset: str = "default", 
                      filter_count: int = 8, sample_rate: float = 48000,
                      intermediate_results_interval: int = 0,
                      progress_callback: Optional[Callable[[OptimizationStep], None]] = None) -> str:
    """
    Start an EQ optimization in a background thread.
    
    Args:
        intermediate_results_interval: Return intermediate results every n filters (0=disabled)
    
    Returns:
        optimization_id: Unique ID for tracking the optimization
    """
    optimizer = RoomEQOptimizer(progress_callback)
    optimization_id = optimizer.optimization_id
    
    _active_optimizations[optimization_id] = optimizer
    
    def optimization_thread():
        try:
            result = optimizer.optimize(
                frequencies, magnitudes, target_curve, optimizer_preset,
                filter_count, sample_rate, intermediate_results_interval
            )
            _optimization_results[optimization_id] = result
        finally:
            if optimization_id in _active_optimizations:
                del _active_optimizations[optimization_id]
    
    thread = threading.Thread(target=optimization_thread)
    thread.daemon = True
    thread.start()
    
    return optimization_id


def get_optimization_status(optimization_id: str) -> Dict[str, Any]:
    """Get current status of an optimization."""
    if optimization_id in _active_optimizations:
        optimizer = _active_optimizations[optimization_id]
        latest_step = optimizer.progress_reporter.steps[-1] if optimizer.progress_reporter.steps else None
        return {
            "status": "running",
            "optimization_id": optimization_id,
            "latest_step": asdict(latest_step) if latest_step else None,
            "step_count": len(optimizer.progress_reporter.steps)
        }
    elif optimization_id in _optimization_results:
        result = _optimization_results[optimization_id]
        return {
            "status": "completed" if result.success else "failed",
            "optimization_id": optimization_id,
            "result": asdict(result)
        }
    else:
        return {
            "status": "not_found",
            "optimization_id": optimization_id
        }


def cancel_optimization(optimization_id: str) -> bool:
    """Cancel a running optimization."""
    if optimization_id in _active_optimizations:
        _active_optimizations[optimization_id].cancel()
        return True
    return False


def get_optimization_result(optimization_id: str) -> Optional[OptimizationResult]:
    """Get completed optimization result."""
    return _optimization_results.get(optimization_id)
