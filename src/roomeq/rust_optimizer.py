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
            
            # Convert curve format for Rust optimizer
            # Extract frequency and magnitude points from curve definition
            frequencies = []
            magnitudes_db = []
            
            for point in curve_data["curve"]:
                frequencies.append(point["frequency"])
                magnitudes_db.append(point["target_db"])
            
            return {
                "name": curve_data["name"],
                "description": curve_data.get("description"),  # Optional field
                "frequencies": frequencies,
                "magnitudes_db": magnitudes_db
            }
            
        except ValueError as e:
            logger.warning(f"Target curve '{target_curve}' not found: {e}")
            logger.debug("Falling back to flat curve")
            # Fallback to flat curve
            return {
                "name": "Flat Response",
                "description": "Flat frequency response (fallback)",
                "frequencies": [20, 20000],
                "magnitudes_db": [0, 0]
            }
    
    def _get_optimizer_params(self, preset: str, add_highpass: bool) -> Dict[str, Any]:
        """Get optimizer parameters by preset name."""
        logger.debug(f"Loading optimizer parameters for preset: '{preset}', add_highpass={add_highpass}")
        
        try:
            # Use centralized presets from presets.py
            preset_data = get_optimizer_preset(preset or "default")
            logger.debug(f"Loaded optimizer preset '{preset_data['name']}' from presets")
            
            # Convert preset format for Rust optimizer (if needed)
            # Keep the preset structure but potentially add description as optional
            params = {
                "name": preset_data["name"],
                "description": preset_data.get("description"),  # Optional field
                "qmax": preset_data["qmax"],
                "mindb": preset_data["mindb"], 
                "maxdb": preset_data["maxdb"],
                "add_highpass": add_highpass if add_highpass is not None else preset_data["add_highpass"],
                "acceptable_error": preset_data["acceptable_error"]
            }
            
            logger.debug(f"Using optimizer params: qmax={params['qmax']}, mindb={params['mindb']}, maxdb={params['maxdb']}")
            return params
            
        except ValueError as e:
            logger.warning(f"Optimizer preset '{preset}' not found: {e}")
            logger.debug("Falling back to default preset")
            # Fallback to default
            default_preset = get_optimizer_preset("default")
            return {
                "name": "Default (fallback)",
                "description": "Default optimization parameters (fallback)",
                "qmax": default_preset["qmax"],
                "mindb": default_preset["mindb"],
                "maxdb": default_preset["maxdb"], 
                "add_highpass": add_highpass if add_highpass is not None else default_preset["add_highpass"],
                "acceptable_error": default_preset["acceptable_error"]
            }
    
    def _run_optimizer_streaming(self, job_file: str) -> Iterator[Dict[str, Any]]:
        """Run the Rust optimizer with streaming output."""
        cmd = [self.binary_path, "--progress", "--result", "--frequency-response"]
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
            
            # Process output line by line
            error_detected = False
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
                all_output_lines.append(line)  # Store all lines for error diagnosis
                
                # Log all output lines for debugging (especially important for errors)
                logger.info(f"Rust optimizer output line {lines_processed}: {line}")
                
                # Check if process exited early
                if process.poll() is not None and process.returncode != 0:
                    logger.warning(f"Process exited early with code {process.returncode} after {lines_processed} lines")
                    logger.error(f"Last output line before failure: {line}")
                    error_detected = True
                
                # Try to parse as JSON first (new combined format)
                try:
                    data = json.loads(line)
                    logger.debug(f"Parsed JSON data with keys: {list(data.keys())}")
                    
                    # Check if it's an error message
                    if data.get("type") == "error":
                        error_message = data.get("message", "Unknown error from Rust optimizer")
                        logger.error(f"Rust optimizer reported error: {error_message}")
                        yield {
                            "type": "error",
                            "optimization_id": optimization_id,
                            "message": f"Rust optimizer error: {error_message}",
                            "details": data,
                            "timestamp": time.time()
                        }
                        # Continue processing to see if there are more details, don't break here
                    
                    # Check if it's a step with optional frequency response
                    elif "step" in data and "filters" in data:
                        # Update our filter list to match the step data
                        if len(data["filters"]) > len(filters):
                            # New filter added
                            new_filter = data["filters"][-1]  # Get the newest filter
                            filters = data["filters"].copy()  # Update our filter list
                            
                            filter_event = {
                                "type": "filter_added", 
                                "optimization_id": optimization_id,
                                "step": step_number,
                                "message": data.get("message", f"Added filter at step {data['step']}"),
                                "filter": new_filter,
                                "total_filters": len(filters),
                                "current_filter_set": filters.copy(),
                                "progress": data.get("progress_percent", 0.0),
                                "timestamp": time.time()
                            }
                            
                            # Add frequency response if included
                            if "frequency_response" in data:
                                filter_event["frequency_response"] = {
                                    "frequencies": data["frequency_response"]["frequencies"],
                                    "magnitude_db": data["frequency_response"]["magnitude_db"],
                                    "phase_degrees": data["frequency_response"].get("phase_degrees", [])
                                }
                            
                            yield filter_event
                            step_number += 1
                    
                    # Check if it's a standalone frequency response (final result)
                    elif data.get("event") == "frequency_response_final":
                        yield {
                            "type": "frequency_response",
                            "optimization_id": optimization_id, 
                            "step": -1,
                            "message": "Final frequency response",
                            "current_filter_set": data.get("current_filter_set", filters.copy()),
                            "total_filters": len(data.get("current_filter_set", filters)),
                            "frequency_response": {
                                "frequencies": data["frequencies"],
                                "magnitude_db": data["magnitude_db"], 
                                "phase_degrees": data.get("phase_degrees", [])
                            },
                            "timestamp": time.time()
                        }
                    
                    continue  # Successfully parsed as JSON, skip legacy parsing
                    
                except (json.JSONDecodeError, KeyError):
                    # Not JSON or missing required fields, try legacy parsing
                    logger.debug(f"Line is not valid JSON, trying legacy parsing: {line[:50]}...")
                    pass
                # Legacy parsing for non-JSON output
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
                                    "current_filter_set": filters.copy(),  # Full list of all filters up to this point
                                    "progress": min(95.0, (len(filters) / 10) * 100),  # Estimate progress
                                    "timestamp": time.time()
                                }
                                step_number += 1
                    except (ValueError, IndexError):
                        continue
                
                elif line.startswith("FREQUENCY_RESPONSE:"):
                    # Parse frequency response data from Rust optimizer
                    try:
                        # Expected format: "FREQUENCY_RESPONSE:step_N:{json_data}"
                        parts = line.split(":", 2)
                        if len(parts) >= 3:
                            step_info = parts[1]  # e.g., "step_0"
                            response_data = json.loads(parts[2])
                            
                            # Extract step number from step_info
                            step_match = step_info.replace('step_', '')
                            try:
                                response_step = int(step_match)
                            except ValueError:
                                response_step = step_number
                            
                            yield {
                                "type": "frequency_response", 
                                "optimization_id": optimization_id,
                                "step": response_step,
                                "message": f"Frequency response calculated after step {response_step + 1}",
                                "current_filter_set": filters.copy(),  # Include current filter set
                                "total_filters": len(filters),
                                "frequency_response": {
                                    "frequencies": response_data["frequencies"],
                                    "magnitude_db": response_data["magnitude_db"],
                                    "phase_degrees": response_data.get("phase_degrees", [])
                                },
                                "timestamp": time.time()
                            }
                    except (ValueError, json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Could not parse frequency response data: {e}")
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
                
                # If we detected an error and the process has finished, break early
                if error_detected and process.poll() is not None:
                    logger.warning("Breaking output processing loop due to detected error and process termination")
                    break
            
            # Check for errors
            return_code = process.poll()  # Non-blocking check
            if return_code is None:
                return_code = process.wait()  # Wait for completion
                
            logger.info(f"Process completed with return code: {return_code}")
            
            if return_code != 0:
                # Try to read any remaining stderr
                stderr_output = ""
                try:
                    if process.stderr:
                        stderr_output = process.stderr.read() or ""
                except Exception as stderr_error:
                    logger.warning(f"Failed to read stderr: {stderr_error}")
                    
                logger.error(f"Optimizer failed with code {return_code}")
                logger.error(f"Stderr output: '{stderr_output}'")
                logger.error(f"Total lines processed: {lines_processed}")
                logger.error(f"All stdout output lines received:")
                for i, output_line in enumerate(all_output_lines, 1):
                    logger.error(f"  Line {i}: {output_line}")
                logger.error(f"Command was: {' '.join(cmd)}")
                logger.error(f"Job file: {job_file}")
                
                # Try to get more diagnostic info by testing the binary
                try:
                    logger.info("Testing Rust binary with --help to check if it's working...")
                    help_result = subprocess.run([self.binary_path, "--help"], 
                                               capture_output=True, text=True, timeout=5)
                    logger.info(f"Binary help test - return code: {help_result.returncode}")
                    if help_result.stdout:
                        logger.info(f"Binary help stdout: {help_result.stdout[:200]}...")
                    if help_result.stderr:
                        logger.info(f"Binary help stderr: {help_result.stderr[:200]}...")
                except Exception as help_error:
                    logger.error(f"Failed to test binary with --help: {help_error}")
                
                # Try to validate the job file content
                try:
                    logger.info("Checking job file content for diagnostic purposes...")
                    with open(job_file, 'r') as f:
                        job_content = f.read()
                    logger.info(f"Job file size: {len(job_content)} bytes")
                    
                    # Parse and validate JSON
                    job_data = json.loads(job_content)
                    logger.info(f"Job data keys: {list(job_data.keys())}")
                    if 'measured_curve' in job_data:
                        measured = job_data['measured_curve']
                        logger.info(f"Measured curve: {len(measured.get('frequencies', []))} frequencies")
                    if 'target_curve' in job_data:
                        target = job_data['target_curve'] 
                        logger.info(f"Target curve: {len(target.get('frequencies', []))} frequencies")
                    logger.info(f"Filter count: {job_data.get('filter_count', 'unknown')}")
                    logger.info(f"Sample rate: {job_data.get('sample_rate', 'unknown')}")
                except Exception as job_error:
                    logger.error(f"Failed to validate job file: {job_error}")
                
                # Extract actual error messages from output lines
                rust_error_messages = []
                for output_line in all_output_lines:
                    try:
                        # Try to parse as JSON to extract error messages
                        line_data = json.loads(output_line)
                        if line_data.get("type") == "error":
                            error_msg = line_data.get("message", output_line)
                            rust_error_messages.append(error_msg)
                    except json.JSONDecodeError:
                        # If not JSON, check if line contains error keywords (but exclude normal status messages)
                        line_lower = output_line.lower().strip()
                        if (any(error_word in line_lower for error_word in ['error:', 'failed:', 'panic!', 'abort:', 'fatal:']) 
                            and not line_lower.startswith('step ') 
                            and not 'progress:' in line_lower):
                            rust_error_messages.append(output_line)
                
                # Construct comprehensive error message
                if rust_error_messages:
                    error_details = "; ".join(rust_error_messages)
                    error_msg = f"Rust optimizer failed with code {return_code}: {error_details}"
                elif stderr_output.strip():
                    error_msg = f"Rust optimizer failed with code {return_code}: {stderr_output.strip()}"
                else:
                    error_msg = f"Rust optimizer failed with code {return_code} (no error message available)"
                    
                logger.error(f"Final error message: {error_msg}")
                raise RustOptimizerError(error_msg)
            else:
                logger.info(f"Optimization completed successfully after processing {lines_processed} lines")
        
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
    logger.info(f"optimize_with_rust called with: frequencies={len(frequencies)} points, "
               f"target_curve='{target_curve}', optimizer_preset='{optimizer_preset}', "
               f"filter_count={filter_count}, sample_rate={sample_rate}, add_highpass={add_highpass}")
    
    # Adjust filter count if high-pass is included
    if add_highpass and filter_count > 1:
        # Rust optimizer automatically adds HP, so we need one less PEQ
        effective_filter_count = filter_count - 1
        logger.info(f"Adjusting filter count from {filter_count} to {effective_filter_count} PEQs + 1 HP")
    else:
        effective_filter_count = filter_count
        logger.info(f"Using {effective_filter_count} filters (no high-pass adjustment needed)")
    
    try:
        logger.info("Starting rust optimizer streaming...")
        step_count = 0
        for step in rust_optimizer.optimize_streaming(
            frequencies=frequencies,
            magnitudes=magnitudes,
            target_curve=target_curve,
            optimizer_preset=optimizer_preset,
            filter_count=effective_filter_count,
            sample_rate=sample_rate,
            add_highpass=add_highpass
        ):
            step_count += 1
            logger.debug(f"Yielding step {step_count}: {step.get('type', 'unknown')} - {step.get('message', 'no message')}")
            yield step
        
        logger.info(f"Rust optimization completed after {step_count} steps")
        
    except Exception as e:
        logger.error(f"Error in optimize_with_rust: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
