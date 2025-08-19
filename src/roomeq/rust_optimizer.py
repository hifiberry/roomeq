"""
Rust-based room EQ optimizer integration with streaming output.

This module provides integration with the high-performance Rust optimizer,
streaming results directly to clients without buffering.
"""

import json
import subprocess
import os
import logging
import time
import uuid
import tempfile
from typing import List, Dict, Any, Optional, Iterator, Tuple
from pathlib import Path

from .presets import get_target_curve, get_optimizer_preset

logger = logging.getLogger(__name__)


class RustOptimizerError(Exception):
    """Exception raised by Rust optimizer integration."""
    pass


class RustOptimizer:
    """Integration with the Rust room EQ optimizer."""
    
    def __init__(self):
        # Find the Rust binary
        self.binary_path = self._find_rust_binary()
        logger.info(f"Using Rust optimizer at: {self.binary_path}")
    
    def _find_rust_binary(self) -> str:
        """Find the Rust optimizer binary."""
        # Try different possible locations
        possible_paths = [
            # Development build
            os.path.join(os.path.dirname(__file__), "..", "rust", "target", "release", "roomeq-optimizer"),
            # Installed package
            "/usr/local/bin/roomeq-optimizer",
            "/usr/bin/roomeq-optimizer",
            # Current directory
            "./roomeq-optimizer"
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
        
        raise RustOptimizerError(
            f"Rust optimizer binary not found. Tried: {possible_paths}. "
            "Please build the Rust optimizer with 'cargo build --release'"
        )
    
    def optimize_streaming(
        self,
        frequencies: List[float],
        magnitudes: List[float],
        target_curve: str = "weighted_flat",
        optimizer_preset: str = "default",
        filter_count: int = 8,
        sample_rate: float = 48000,
        add_highpass: bool = True
    ) -> Iterator[Dict[str, Any]]:
        """
        Run optimization with streaming output.
        
        Yields optimization steps in real-time without buffering.
        """
        # Create job configuration
        job_config = self._create_job_config(
            frequencies, magnitudes, target_curve, optimizer_preset, 
            filter_count, sample_rate, add_highpass
        )
        
        # Create temporary file for job configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(job_config, f, indent=2)
            job_file = f.name
        
        try:
            # Run optimizer with streaming output
            yield from self._run_optimizer_streaming(job_file)
        finally:
            # Clean up temporary file
            try:
                os.unlink(job_file)
            except:
                pass
    
    def _create_job_config(
        self,
        frequencies: List[float],
        magnitudes: List[float],
        target_curve: str,
        optimizer_preset: str,
        filter_count: int,
        sample_rate: float,
        add_highpass: bool
    ) -> Dict[str, Any]:
        """Create job configuration for Rust optimizer."""
        # Convert target curve name to curve data
        target_curve_data = self._get_target_curve_data(target_curve)
        
        # Convert optimizer preset to parameters
        optimizer_params = self._get_optimizer_params(optimizer_preset, add_highpass)
        
        return {
            "measured_curve": {
                "frequencies": frequencies,
                "magnitudes_db": magnitudes
            },
            "target_curve": target_curve_data,
            "optimizer_params": optimizer_params,
            "filter_count": filter_count,
            "sample_rate": sample_rate
        }
    
    def _get_target_curve_data(self, target_curve: str) -> Dict[str, Any]:
        """Convert target curve name to curve data."""
        try:
            return get_target_curve(target_curve)
        except ValueError:
            # Fallback to weighted_flat for unknown curves
            logger.warning(f"Unknown target curve '{target_curve}', using 'weighted_flat'")
            return get_target_curve("weighted_flat")
    
    def _get_optimizer_params(self, preset: str, add_highpass: bool) -> Dict[str, Any]:
        """Convert optimizer preset to parameters."""
        try:
            params = get_optimizer_preset(preset)
            # Override the add_highpass setting if explicitly specified
            params["add_highpass"] = add_highpass
            return params
        except ValueError:
            # Fallback to default preset for unknown presets
            logger.warning(f"Unknown optimizer preset '{preset}', using 'default'")
            params = get_optimizer_preset("default")
            params["add_highpass"] = add_highpass
            return params
    
    def _run_optimizer_streaming(self, job_file: str) -> Iterator[Dict[str, Any]]:
        """Run the Rust optimizer with streaming output."""
        cmd = [self.binary_path, "--progress", "--result", "--human-readable"]
        
        try:
            # Start process with streaming I/O
            with open(job_file, 'r') as f:
                process = subprocess.Popen(
                    cmd,
                    stdin=f,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1  # Line buffered
                )
            
            optimization_id = str(uuid.uuid4())[:8]
            step_number = 0
            start_time = time.time()
            
            # Track filters and results
            filters = []
            usable_range = None
            final_result = None
            
            # Process output line by line
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # Parse different types of output
                if line.startswith("Usable frequency range:"):
                    # Extract frequency range
                    parts = line.split()
                    try:
                        f_low = float(parts[3])
                        f_high = float(parts[6])
                        candidates = int(parts[8].rstrip(')'))
                        usable_range = (f_low, f_high, candidates)
                        
                        yield {
                            "type": "initialization",
                            "optimization_id": optimization_id,
                            "step": step_number,
                            "message": f"Detected usable frequency range: {f_low:.1f} - {f_high:.1f} Hz ({candidates} candidates)",
                            "usable_range": {"f_low": f_low, "f_high": f_high, "candidates": candidates},
                            "progress": 0.0,
                            "timestamp": time.time()
                        }
                        step_number += 1
                    except (ValueError, IndexError):
                        continue
                
                elif line.startswith("Step "):
                    # Parse optimization step
                    try:
                        # Example: "Step 1: Added filter 1 at 16000.0Hz, Q=0.5, 3.0dB (Error: 2.62 dB, Progress: 0.0%)"
                        if "Added" in line and "filter" in line:
                            parts = line.split()
                            step_idx = int(parts[1].rstrip(':'))
                            
                            # Extract filter details
                            filter_info = self._parse_filter_line(line)
                            if filter_info:
                                filters.append(filter_info)
                                
                                yield {
                                    "type": "filter_added",
                                    "optimization_id": optimization_id,
                                    "step": step_number,
                                    "message": line,
                                    "filter": filter_info,
                                    "total_filters": len(filters),
                                    "progress": min(95.0, (len(filters) / 10) * 100),  # Estimate progress
                                    "timestamp": time.time()
                                }
                                step_number += 1
                    except (ValueError, IndexError):
                        continue
                
                elif "Added:" in line:
                    # Filter format line: "  Added: eq:16000.0:0.500:3.00"
                    try:
                        filter_str = line.split("Added:")[1].strip()
                        filter_parts = filter_str.split(':')
                        
                        if len(filter_parts) >= 4 and filter_parts[0] in ['eq', 'hp']:
                            filter_type = "high_pass" if filter_parts[0] == 'hp' else "peaking_eq"
                            frequency = float(filter_parts[1])
                            q = float(filter_parts[2])
                            gain_db = float(filter_parts[3]) if len(filter_parts) > 3 else 0.0
                            
                            # Update the last filter with detailed info
                            if filters:
                                filters[-1].update({
                                    "text_format": filter_str,
                                    "coefficients": None  # Could calculate if needed
                                })
                    except (ValueError, IndexError):
                        continue
                
                elif line.startswith("=== OPTIMIZATION RESULTS ==="):
                    # Final results section - start parsing
                    final_result = self._parse_final_results(process.stdout, filters, usable_range)
                    
                    if final_result:
                        yield {
                            "type": "completed",
                            "optimization_id": optimization_id,
                            "step": step_number,
                            "message": "Optimization completed successfully",
                            "result": final_result,
                            "progress": 100.0,
                            "processing_time": time.time() - start_time,
                            "timestamp": time.time()
                        }
                    break
            
            # Check for errors
            return_code = process.wait()
            if return_code != 0:
                stderr_output = process.stderr.read()
                raise RustOptimizerError(f"Optimizer failed with code {return_code}: {stderr_output}")
        
        except subprocess.SubprocessError as e:
            raise RustOptimizerError(f"Failed to run optimizer: {e}")
        except Exception as e:
            raise RustOptimizerError(f"Optimization error: {e}")
    
    def _parse_filter_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a filter line to extract filter information."""
        try:
            # Example: "Step 1: Added filter 1 at 16000.0Hz, Q=0.5, 3.0dB (Error: 2.62 dB, Progress: 0.0%)"
            parts = line.split()
            
            # Find frequency
            freq_part = [p for p in parts if 'Hz' in p][0]
            frequency = float(freq_part.replace('Hz,', '').replace('Hz', ''))
            
            # Find Q
            q_part = [p for p in parts if p.startswith('Q=')][0]
            q = float(q_part.replace('Q=', '').rstrip(','))
            
            # Find gain
            gain_parts = [p for p in parts if 'dB' in p and '(' not in p]
            if gain_parts:
                gain_db = float(gain_parts[0].replace('dB', ''))
            else:
                gain_db = 0.0
            
            # Determine filter type
            if "high-pass" in line.lower():
                filter_type = "high_pass"
            else:
                filter_type = "peaking_eq"
            
            return {
                "filter_type": filter_type,
                "frequency": frequency,
                "q": q,
                "gain_db": gain_db,
                "description": f"{filter_type.replace('_', ' ').title()} {frequency}Hz {gain_db:+.1f}dB"
            }
        except (ValueError, IndexError, AttributeError):
            return None
    
    def _parse_final_results(self, stdout, filters: List[Dict], usable_range: Optional[Tuple]) -> Optional[Dict[str, Any]]:
        """Parse the final results section."""
        try:
            result = {
                "success": False,
                "filters": filters,
                "filter_count": len(filters),
                "usable_range": usable_range,
                "original_error": None,
                "final_error": None,
                "improvement_db": None,
                "processing_time": None
            }
            
            # Read lines until we get the key metrics
            for line in stdout:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("Success:"):
                    result["success"] = "true" in line.lower()
                elif line.startswith("Original error:"):
                    try:
                        error_str = line.split(":")[1].strip().replace("dB RMS", "")
                        result["original_error"] = float(error_str)
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("Final error:"):
                    try:
                        error_str = line.split(":")[1].strip().replace("dB RMS", "")
                        result["final_error"] = float(error_str)
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("Improvement:"):
                    try:
                        improvement_str = line.split(":")[1].strip().replace("dB", "")
                        result["improvement_db"] = float(improvement_str)
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("Processing time:"):
                    try:
                        time_str = line.split(":")[1].strip().replace("ms", "")
                        result["processing_time"] = float(time_str) / 1000.0  # Convert to seconds
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("Filters:"):
                    # Filter details follow - we already have them
                    break
            
            return result
        except Exception as e:
            logger.error(f"Error parsing final results: {e}")
            return None


# Global instance
rust_optimizer = RustOptimizer()


def optimize_with_rust(
    frequencies: List[float],
    magnitudes: List[float],
    target_curve: str = "weighted_flat",
    optimizer_preset: str = "default",
    filter_count: int = 8,
    sample_rate: float = 48000,
    add_highpass: bool = True
) -> Iterator[Dict[str, Any]]:
    """
    Run room EQ optimization using the Rust backend with streaming output.
    
    This function streams optimization steps in real-time without buffering,
    allowing clients to receive immediate feedback on progress.
    
    Args:
        frequencies: Frequency points in Hz
        magnitudes: Magnitude values in dB
        target_curve: Target response curve name
        optimizer_preset: Optimization preset name
        filter_count: Number of filters to generate
        sample_rate: Audio sample rate
        add_highpass: Whether to include adaptive high-pass filter
    
    Yields:
        Optimization steps with real-time progress information
    """
    # Adjust filter count if high-pass is included
    if add_highpass and filter_count > 1:
        # Rust optimizer automatically adds HP, so we need one less PEQ
        effective_filter_count = filter_count - 1
        logger.info(f"Adjusting filter count from {filter_count} to {effective_filter_count} PEQs + 1 HP")
    else:
        effective_filter_count = filter_count
    
    yield from rust_optimizer.optimize_streaming(
        frequencies=frequencies,
        magnitudes=magnitudes,
        target_curve=target_curve,
        optimizer_preset=optimizer_preset,
        filter_count=effective_filter_count,
        sample_rate=sample_rate,
        add_highpass=add_highpass
    )
