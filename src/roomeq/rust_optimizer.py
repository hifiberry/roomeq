"""
Simple pass-through integration with the Rust room EQ optimizer.

This module streams output directly from the Rust optimizer without any parsing or processing.
"""

import json
import subprocess
import os
import time
import logging
import tempfile
from typing import Dict, Any, Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


class RustOptimizerError(Exception):
    """Exception raised when the Rust optimizer fails."""
    pass


class RustOptimizer:
    """Simple pass-through integration with the Rust room EQ optimizer."""

    def __init__(self):
        """Initialize the Rust optimizer integration."""
        self.binary_path = self._find_binary()
        logger.info(f"Using Rust optimizer at: {self.binary_path}")

    def _find_binary(self) -> str:
        """Find the Rust optimizer binary."""
        possible_paths = [
            "/usr/bin/roomeq-optimizer",
            "/usr/local/bin/roomeq-optimizer",
            Path(__file__).parent.parent.parent / "rust" / "target" / "release" / "roomeq-optimizer",
            Path(__file__).parent.parent.parent / "rust" / "target" / "debug" / "roomeq-optimizer"
        ]
        
        for path in possible_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                logger.info(f"Found executable Rust optimizer binary: {path}")
                return str(path)
        
        raise RustOptimizerError(
            f"Rust optimizer binary not found. Tried: {possible_paths}. "
            "Please build the Rust optimizer with 'cargo build --release'"
        )

    def optimize_streaming_json(self, job_data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Run optimization with raw line-by-line streaming output.
        
        Args:
            job_data: Complete optimization job JSON object
            
        Yields:
            Raw output lines from the Rust optimizer
        """
        logger.info("Starting Rust optimization with direct JSON pass-through")
        start_time = time.time()
        
        # Create temporary job file
        job_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(job_data, f, indent=2)
                job_file = f.name
            
            # Run the Rust optimizer
            cmd = [self.binary_path, "--progress", "--frequency-response"]
            logger.info(f"Executing command: {' '.join(cmd)}")
            logger.info(f"Using job file: {job_file}")
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            logger.info(f"Subprocess started with PID: {process.pid}")
            
            # Send job data to stdin
            with open(job_file, 'r') as f:
                job_data_str = f.read()
            
            process.stdin.write(job_data_str)
            process.stdin.flush()
            process.stdin.close()
            
            # Stream output line by line
            lines_processed = 0
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    logger.info(f"Process finished. Processed {lines_processed} lines total.")
                    break
                
                if not line:
                    continue
                
                line = line.strip()
                if not line:
                    continue
                    
                lines_processed += 1
                logger.info(f"Rust optimizer output line {lines_processed}: {line}")
                
                # Check if process exited with error
                if process.poll() is not None and process.returncode != 0:
                    logger.warning(f"Process exited early with code {process.returncode}")
                    break
                
                # Yield raw line
                yield {
                    "type": "output",
                    "line": line,
                    "line_number": lines_processed
                }
            
            # Wait for process completion
            return_code = process.poll()
            if return_code is None:
                return_code = process.wait()
                
            logger.info(f"Process completed with return code: {return_code}")
            
            if return_code == 0:
                logger.info(f"Optimization completed successfully after processing {lines_processed} lines")
                yield {
                    "type": "completed",
                    "message": "Optimization completed successfully",
                    "lines_processed": lines_processed,
                    "processing_time": time.time() - start_time
                }
            else:
                # Read stderr for error details
                stderr_output = ""
                try:
                    if process.stderr:
                        stderr_output = process.stderr.read() or ""
                except Exception as e:
                    logger.warning(f"Failed to read stderr: {e}")
                    
                logger.error(f"Optimizer failed with code {return_code}")
                if stderr_output:
                    logger.error(f"Stderr: {stderr_output}")
                
                error_msg = f"Rust optimizer failed with code {return_code}"
                if stderr_output.strip():
                    error_msg += f": {stderr_output.strip()}"
                    
                raise RustOptimizerError(error_msg)
                
        except Exception as e:
            logger.error(f"Error during streaming optimization: {e}")
            raise RustOptimizerError(f"Optimization failed: {e}")
        finally:
            # Clean up temporary file
            if job_file and os.path.exists(job_file):
                try:
                    os.unlink(job_file)
                    logger.debug(f"Cleaned up temporary job file: {job_file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up job file {job_file}: {e}")
                    
        total_time = time.time() - start_time
        logger.info(f"EQ optimization completed in {total_time:.2f} seconds")


def optimize_with_rust_json(job_data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Run room EQ optimization using the Rust backend with streaming output.
    
    Args:
        job_data: Complete optimization job JSON object
    
    Yields:
        Raw output lines from the Rust optimizer
    """
    logger.info(f"optimize_with_rust_json called with job data keys: {list(job_data.keys())}")
    
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
