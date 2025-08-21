"""
Rust-based room EQ optimizer integration with streaming output.

This module provides integration with the high-performance Rust optimizer,
streaming results directly to clients without buffering.
"""

import json
import subprocess
import os
import time
import logging
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
        
        logger.debug(f"Searching for Rust optimizer binary in {len(possible_paths)} locations...")
        
        for i, path in enumerate(possible_paths):
            logger.debug(f"Checking path {i+1}/{len(possible_paths)}: {path}")
            if os.path.exists(path):
                logger.debug(f"  Path exists: {path}")
                if os.access(path, os.X_OK):
                    logger.info(f"Found executable Rust optimizer binary: {path}")
                    return path
                else:
                    logger.warning(f"  Path exists but is not executable: {path}")
            else:
                logger.debug(f"  Path does not exist: {path}")
        
        logger.error("No Rust optimizer binary found in any expected location")
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
        logger.info(f"Starting Rust optimization: {len(frequencies)} freq points, "
                   f"target={target_curve}, preset={optimizer_preset}, "
                   f"filters={filter_count}, sample_rate={sample_rate}, hp={add_highpass}")
        
        start_time = time.time()
        logger.info(f"EQ optimization started at {time.strftime('%H:%M:%S')}")
        
        # Create job configuration
        config_start = time.time()
        logger.debug("Creating job configuration...")
        job_config = self._create_job_config(
            frequencies, magnitudes, target_curve, optimizer_preset, 
            filter_count, sample_rate, add_highpass
        )
        config_time = time.time() - config_start
        logger.debug(f"Job config created in {config_time:.3f}s: {len(job_config['measured_curve']['frequencies'])} frequencies, "
                    f"filter_count={job_config['filter_count']}")
        
        # Create temporary file for job configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(job_config, f, indent=2)
            job_file = f.name
        
        logger.info(f"Job configuration written to: {job_file}")
        
        try:
            # Run optimizer with streaming output
            stream_start = time.time()
            logger.info("Starting optimizer process...")
            yield from self._run_optimizer_streaming(job_file)
            stream_time = time.time() - stream_start
            total_time = time.time() - start_time
            logger.info(f"Optimizer process completed successfully in {total_time:.3f}s (stream: {stream_time:.3f}s)")
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"Optimizer process failed after {error_time:.3f}s: {e}")
            raise
        finally:
            # Clean up temporary file
            try:
                os.unlink(job_file)
                logger.debug(f"Cleaned up job file: {job_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up job file {job_file}: {e}")
    
    def optimize_streaming_json(self, job_data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Run optimization with streaming output using a complete job JSON object.
        
        This method accepts a complete job definition and passes it directly
        to the Rust optimizer without any processing or transformation.
        
        Args:
            job_data: Complete optimization job JSON object
            
        Yields:
            Optimization steps in real-time without buffering.
        """
        logger.info("Starting Rust optimization with direct JSON pass-through")
        
        start_time = time.time()
        logger.info(f"EQ optimization started at {time.strftime('%H:%M:%S')}")
        
        # Create temporary job file with the exact JSON provided
        job_file = None
        try:
            # Write job data directly to temporary file without any modification
            job_file = self._write_job_file(job_data)
            
            # Run optimizer and stream results
            yield from self._run_optimizer_streaming(job_file)
            
        except Exception as e:
            logger.error(f"Error during streaming optimization: {e}")
            raise RustOptimizerError(f"Optimization failed: {e}")
        finally:
            # Clean up temporary job file
            if job_file and os.path.exists(job_file):
                try:
                    os.unlink(job_file)
                    logger.debug(f"Cleaned up temporary job file: {job_file}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up job file {job_file}: {cleanup_error}")
                    
        total_time = time.time() - start_time
        logger.info(f"EQ optimization completed in {total_time:.2f} seconds")

    def _write_job_file(self, job_data: Dict[str, Any]) -> str:
        """Write job data to a temporary file without any processing."""
        try:
            # Create temporary file for job configuration
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(job_data, f, indent=2)
                job_file = f.name
            
            logger.debug(f"Created job file: {job_file}")
            logger.debug(f"Job data written: {len(json.dumps(job_data))} characters")
            
            return job_file
            
        except Exception as e:
            logger.error(f"Error writing job file: {e}")
            raise RustOptimizerError(f"Failed to create job file: {e}")
    
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
        logger.debug(f"Creating job config with {len(frequencies)} frequencies, target_curve='{target_curve}', preset='{optimizer_preset}'")
        
        # Convert target curve name to curve data
        logger.debug("Getting target curve data...")
        target_curve_data = self._get_target_curve_data(target_curve)
        logger.debug(f"Target curve data loaded: {len(target_curve_data.get('frequencies', []))} points")
        
        # Convert optimizer preset to parameters
        logger.debug("Getting optimizer parameters...")
        optimizer_params = self._get_optimizer_params(optimizer_preset, add_highpass)
        logger.debug(f"Optimizer params loaded: {list(optimizer_params.keys())}")
        
        job_config = {
            "name": f"roomeq_optimization_{int(time.time())}",
            "description": f"Room EQ optimization using {optimizer_preset} preset and {target_curve} target curve",
            "measured_curve": {
                "name": "measured_frequency_response",
                "description": f"Measured frequency response with {len(frequencies)} data points",
                "frequencies": frequencies,
                "magnitudes_db": magnitudes
            },
            "target_curve": target_curve_data,
            "optimizer_params": optimizer_params,
            "filter_count": filter_count,
            "sample_rate": sample_rate
        }
        
        logger.debug(f"Job configuration created successfully: measured_curve has {len(job_config['measured_curve']['frequencies'])} points")
        return job_config
    
    def _get_target_curve_data(self, target_curve: str) -> Dict[str, Any]:
        """Get target curve data by name."""
        logger.debug(f"Loading target curve data for: '{target_curve}'")
        
        try:
            # Use centralized presets from presets.py
            curve_data = get_target_curve(target_curve or "flat")
            logger.debug(f"Loaded target curve '{curve_data['name']}' from presets")
            
            # Keep the original curve format for Rust optimizer (with weights)
            # Convert the curve points to the expected CurvePoint format
            curve_points = []
            for point in curve_data["curve"]:
                curve_points.append({
                    "frequency": point["frequency"],
                    "target_db": point["target_db"], 
                    "weight": point.get("weight")  # Can be None, float, or list
                })

            # Only include optional metadata fields if they have meaningful values
            result = {
                "curve": curve_points
            }
            
            # Add optional fields only if they exist and have non-empty values
            if curve_data.get("name"):
                result["name"] = curve_data["name"]
            if curve_data.get("description"):
                result["description"] = curve_data["description"]
            if curve_data.get("expert") is not None:
                result["expert"] = curve_data["expert"]
                
            return result
            
        except ValueError as e:
            logger.warning(f"Target curve '{target_curve}' not found: {e}")
            logger.debug("Falling back to flat curve")
            # Fallback to flat curve - minimal structure without metadata
            return {
                "curve": [
                    {"frequency": 20.0, "target_db": 0.0, "weight": None},
                    {"frequency": 20000.0, "target_db": 0.0, "weight": None}
                ]
            }
            
    def _get_optimizer_params(self, preset: str, add_highpass: bool) -> Dict[str, Any]:
        """Get optimizer parameters by preset name."""
        logger.debug(f"Loading optimizer parameters for preset: '{preset}', add_highpass={add_highpass}")
        
        try:
            # Use centralized presets from presets.py
            preset_data = get_optimizer_preset(preset or "default")
            logger.debug(f"Loaded optimizer preset '{preset_data['name']}' from presets")
            
            # Convert preset format for Rust optimizer - only include functional parameters
            params = {
                "qmax": preset_data["qmax"],
                "mindb": preset_data["mindb"], 
                "maxdb": preset_data["maxdb"],
                "add_highpass": add_highpass if add_highpass is not None else preset_data["add_highpass"],
                "acceptable_error": preset_data["acceptable_error"]
            }
            
            # Add optional metadata fields only if they exist and have meaningful values
            if preset_data.get("name"):
                params["name"] = preset_data["name"]
            if preset_data.get("description"):
                params["description"] = preset_data["description"]
            
            logger.debug(f"Using optimizer params: qmax={params['qmax']}, mindb={params['mindb']}, maxdb={params['maxdb']}")
            return params
            
        except ValueError as e:
            logger.warning(f"Optimizer preset '{preset}' not found: {e}")
            logger.debug("Falling back to default preset")
            # Fallback to default - only functional parameters
            default_preset = get_optimizer_preset("default")
            return {
                "qmax": default_preset["qmax"],
                "mindb": default_preset["mindb"],
                "maxdb": default_preset["maxdb"],
                "add_highpass": add_highpass if add_highpass is not None else default_preset["add_highpass"],
                "acceptable_error": default_preset["acceptable_error"]
            }
    
    def _run_optimizer_streaming(self, job_file: str) -> Iterator[Dict[str, Any]]:
        """Run the Rust optimizer with streaming output."""
        cmd = [self.binary_path, "--progress", "--frequency-response"]
        logger.info(f"Executing command: {' '.join(cmd)}")
        logger.info(f"Using job file: {job_file}")
        
        try:
            # Read job configuration
            logger.debug("Reading job configuration...")
            with open(job_file, 'r') as f:
                job_data = f.read()
            logger.debug(f"Job data loaded: {len(job_data)} characters")
            
            # Validate job data is valid JSON
            try:
                job_config = json.loads(job_data)
                logger.debug(f"Job config validation: {len(job_config.get('measured_curve', {}).get('frequencies', []))} frequencies")
                logger.debug(f"Job config validation: filter_count={job_config.get('filter_count', 'unknown')}")
                logger.debug(f"Job config validation: sample_rate={job_config.get('sample_rate', 'unknown')}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in job file: {e}")
                raise RustOptimizerError(f"Invalid job configuration JSON: {e}")
            
            # Start process with streaming I/O
            logger.debug("Starting subprocess...")
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            logger.info(f"Subprocess started with PID: {process.pid}")
            
            # Send job data to stdin and close it
            logger.debug("Sending job data to subprocess stdin...")
            logger.debug(f"Job data to send (first 500 chars): {job_data[:500]}...")
            try:
                process.stdin.write(job_data)
                process.stdin.flush()  # Make sure data is sent immediately
                process.stdin.close()
                logger.debug("Job data sent and stdin closed")
            except Exception as e:
                logger.error(f"Failed to send job data to subprocess: {e}")
                logger.error(f"Process poll status: {process.poll()}")
                try:
                    process.terminate()
                    logger.info("Terminated subprocess after stdin error")
                except:
                    pass
                raise
            
            optimization_id = str(uuid.uuid4())[:8]
            step_number = 0
            start_time = time.time()
            lines_processed = 0
            
            # Track filters and results
            filters = []
            usable_range = None
            final_result = None
            all_output_lines = []  # Keep track of all output for error diagnosis
            
            logger.info("Starting to process optimizer output...")
            
            # Process output line by line - just pass through without parsing
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    logger.info(f"Process finished, no more output. Processed {lines_processed} lines total.")
                    break
                
                if not line:
                    # Process is still running but no output yet, continue waiting
                    continue
                
                line = line.strip()
                if not line:
                    continue
                    
                lines_processed += 1
                
                # Log all output lines for debugging
                logger.info(f"Rust optimizer output line {lines_processed}: {line}")
                
                # Check if process exited early
                if process.poll() is not None and process.returncode != 0:
                    logger.warning(f"Process exited early with code {process.returncode} after {lines_processed} lines")
                    logger.error(f"Last output line before failure: {line}")
                    break
                
                # Just yield the raw line as a simple message
                yield {
                    "type": "output",
                    "line": line,
                    "line_number": lines_processed
                }
            
            # Wait for process to complete
            return_code = process.poll()
            if return_code is None:
                return_code = process.wait()
                
            logger.info(f"Process completed with return code: {return_code}")
            
            if return_code == 0:
                logger.info(f"Optimization completed successfully after processing {lines_processed} lines")
                yield {
                    "type": "completed",
                    "message": "Optimization completed successfully",
                    "lines_processed": lines_processed
                }
            else:
                # Try to read any remaining stderr
                stderr_output = ""
                try:
                    if process.stderr:
                        stderr_output = process.stderr.read() or ""
                except Exception as stderr_error:
                    logger.warning(f"Failed to read stderr: {stderr_error}")
                    
                logger.error(f"Optimizer failed with code {return_code}")
                if stderr_output:
                    logger.error(f"Stderr output: '{stderr_output}'")
                
                error_msg = f"Rust optimizer failed with code {return_code}"
                if stderr_output.strip():
                    error_msg += f": {stderr_output.strip()}"
                    
                raise RustOptimizerError(error_msg)
        
        except subprocess.SubprocessError as e:
            logger.error(f"Subprocess error: {e}")
            raise RustOptimizerError(f"Failed to run optimizer: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in optimizer: {e}")
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


def optimize_with_rust_json(job_data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Run room EQ optimization using the Rust backend with streaming output.
    
    This function accepts a complete job JSON object and passes it directly
    to the Rust optimizer without any processing or transformation.
    
    Args:
        job_data: Complete optimization job JSON object
    
    Yields:
        Optimization steps with real-time progress information
    """
    logger.info(f"optimize_with_rust_json called with job data keys: {list(job_data.keys())}")
    
    # Create a simple pass-through optimizer instance
    rust_optimizer = RustOptimizer()
    
    try:
        logger.info("Starting rust optimizer streaming with direct JSON...")
        step_count = 0
        for step in rust_optimizer.optimize_streaming_json(job_data):
            step_count += 1
            logger.debug(f"Received step {step_count}: {step.get('type', 'unknown')}")
            yield step
            
        logger.info(f"Rust optimizer streaming completed after {step_count} steps")
        
    except Exception as e:
        logger.error(f"Error in rust optimizer streaming: {e}")
        raise RustOptimizerError(f"Rust optimizer failed: {e}")
