#!/usr/bin/env python3
"""
Example script showing how to use the Rust EQ optimizer.

This script demonstrates how to:
1. Prepare optimization job data in the correct format
2. Send it to the Rust optimizer via stdin 
3. Receive step-by-step progress via stdout
4. Parse the JSON output for each optimization step

Copyright (c) 2025 HiFiBerry
"""

import json
import subprocess
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

def create_sample_measured_curve() -> Dict[str, Any]:
    """Create a sample measured frequency response curve."""
    # Generate log-spaced frequencies from 20Hz to 20kHz
    frequencies = np.logspace(np.log10(20), np.log10(20000), 100).tolist()
    
    # Generate a realistic room response with some peaks and dips
    magnitudes_db = []
    for freq in frequencies:
        # Base response with some room modes
        if freq < 100:
            # Bass boost with room modes
            response = 2.0 * np.sin(freq / 30.0) + np.random.normal(0, 0.5)
        elif freq < 300:
            # Room mode region with nulls and peaks
            response = -3.0 + 4.0 * np.sin(freq / 50.0) + np.random.normal(0, 1.0)
        elif freq < 2000:
            # Midrange with some irregularities
            response = np.random.normal(0, 0.8)
        elif freq < 8000:
            # Presence region
            response = 1.0 + np.random.normal(0, 0.6)
        else:
            # Treble roll-off
            response = -0.5 * np.log10(freq / 8000) + np.random.normal(0, 0.4)
        
        magnitudes_db.append(response)
    
    return {
        "frequencies": frequencies,
        "magnitudes_db": magnitudes_db
    }

def create_target_curve(curve_type: str = "weighted_flat") -> Dict[str, Any]:
    """Create a target curve definition."""
    target_curves = {
        "flat": {
            "name": "Flat Response",
            "description": "Flat frequency response across all frequencies", 
            "expert": False,
            "curve": [
                {"frequency": 20.0, "target_db": 0.0, "weight": None},
                {"frequency": 25000.0, "target_db": 0.0, "weight": None}
            ]
        },
        "weighted_flat": {
            "name": "Weighted Flat",
            "description": "Flat response with frequency-dependent correction weights",
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
        "room_only": {
            "name": "Room Correction Only",
            "description": "Focus correction on room modes (low frequencies up to 250Hz)",
            "expert": False,
            "curve": [
                {"frequency": 20.0, "target_db": 0.0, "weight": [1.0, 0.5]},
                {"frequency": 250.0, "target_db": 0.0, "weight": None}
            ]
        }
    }
    
    return target_curves.get(curve_type, target_curves["flat"])

def create_optimizer_preset(preset_type: str = "default") -> Dict[str, Any]:
    """Create optimizer preset parameters."""
    presets = {
        "default": {
            "name": "Default",
            "description": "Balanced optimization with moderate Q values",
            "qmax": 10.0,
            "mindb": -10.0,
            "maxdb": 3.0,
            "add_highpass": True,
        },
        "smooth": {
            "name": "Smooth", 
            "description": "Moderate correction with medium Q values",
            "qmax": 5.0,
            "mindb": -10.0,
            "maxdb": 3.0,
            "add_highpass": True,
        },
        "aggressive": {
            "name": "Aggressive",
            "description": "Strong correction with high Q values", 
            "qmax": 20.0,
            "mindb": -20.0,
            "maxdb": 6.0,
            "add_highpass": True,
        }
    }
    
    return presets.get(preset_type, presets["default"])

def create_optimization_job(
    curve_type: str = "weighted_flat",
    preset_type: str = "default", 
    filter_count: int = 8,
    sample_rate: float = 48000.0
) -> Dict[str, Any]:
    """Create a complete optimization job."""
    return {
        "measured_curve": create_sample_measured_curve(),
        "target_curve": create_target_curve(curve_type),
        "optimizer_params": create_optimizer_preset(preset_type), 
        "sample_rate": sample_rate,
        "filter_count": filter_count
    }

def run_rust_optimizer(
    job: Dict[str, Any], 
    rust_binary_path: str = "./target/release/roomeq-optimizer",
    show_progress: bool = True,
    show_result: bool = True,
    human_readable: bool = False
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Run the Rust optimizer and collect step outputs."""
    
    # Convert job to JSON
    job_json = json.dumps(job, indent=None, separators=(',', ':'))
    
    print(f"🔧 Starting Rust EQ optimizer...")
    print(f"   Target: {job['target_curve']['name']}")
    print(f"   Preset: {job['optimizer_params']['name']}")
    print(f"   Filters: {job['filter_count']}")
    print(f"   Sample rate: {job['sample_rate']}Hz")
    print(f"   Progress: {show_progress}, Result: {show_result}, Human-readable: {human_readable}")
    print()
    
    # Build command line arguments
    cmd = [rust_binary_path]
    if show_progress:
        cmd.append("--progress")
    if show_result:
        cmd.append("--result")
    if human_readable:
        cmd.append("--human-readable")
    
    # Start the Rust process
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Send job data to stdin
        process.stdin.write(job_json)
        process.stdin.close()
        
        steps = []
        result = None
        
        # Read outputs from stdout
        for line in process.stdout:
            line = line.strip()
            if line:
                if human_readable:
                    # Just print human-readable output directly
                    print(line)
                else:
                    # Parse JSON output
                    try:
                        data = json.loads(line)
                        
                        # Differentiate between step and result JSON
                        if "step" in data and "progress_percent" in data:
                            # This is an optimization step
                            steps.append(data)
                            
                            # Print progress
                            print(f"Step {data['step']}: {data['message']}")
                            print(f"  Filters: {len(data['filters'])}")
                            print(f"  Error: {data['residual_error']:.2f} dB RMS")
                            print(f"  Progress: {data['progress_percent']:.1f}%")
                            
                            if data['filters']:
                                last_filter = data['filters'][-1]
                                print(f"  Last filter: {last_filter.get('description', 'Unknown')}")
                            print()
                        elif "success" in data and "final_error" in data:
                            # This is the final result
                            result = data
                            print("=== FINAL RESULT ===")
                            print(f"Success: {data['success']}")
                            print(f"Original error: {data['original_error']:.2f} dB RMS")
                            print(f"Final error: {data['final_error']:.2f} dB RMS") 
                            print(f"Improvement: {data['improvement_db']:.1f} dB")
                            print(f"Processing time: {data['processing_time_ms']} ms")
                            print(f"Filters created: {len(data['filters'])}")
                            print()
                        
                    except json.JSONDecodeError:
                        print(f"Note: {line}")  # Non-JSON output
                        continue
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Read any messages from stderr
        stderr_output = process.stderr.read()
        if stderr_output:
            print("=== Additional Info ===")
            print(stderr_output)
        
        if return_code != 0:
            print(f"❌ Optimizer failed with return code {return_code}")
            return [], None
        
        print(f"✅ Optimization completed successfully!")
        return steps, result
        
    except FileNotFoundError:
        print(f"❌ Rust binary not found: {rust_binary_path}")
        print("   Make sure to compile the Rust code first:")
        print("   cd src/rust && cargo build --release")
        return [], None
    except Exception as e:
        print(f"❌ Error running optimizer: {e}")
        return [], None

def analyze_results(steps: List[Dict[str, Any]], result: Optional[Dict[str, Any]] = None):
    """Analyze and summarize optimization results."""
    if not steps and not result:
        print("No optimization steps or results to analyze.")
        return
    
    print("\n=== Optimization Analysis ===")
    
    # Use result data if available, otherwise fallback to steps
    if result:
        initial_error = result.get('original_error', 0)
        final_error = result.get('final_error', 0) 
        improvement_db = result.get('improvement_db', 0)
        filters = result.get('filters', [])
        processing_time = result.get('processing_time_ms', 0)
        
        print(f"Initial error: {initial_error:.2f} dB RMS")
        print(f"Final error: {final_error:.2f} dB RMS") 
        print(f"Improvement: {improvement_db:.1f} dB")
        print(f"Processing time: {processing_time} ms")
        print(f"Total filters: {len(filters)}")
    elif steps:
        first_step = steps[0]
        last_step = steps[-1]
        
        initial_error = first_step['residual_error']
        final_error = last_step['residual_error']
        improvement_db = 20 * np.log10(initial_error / max(final_error, 1e-10))
        filters = last_step.get('filters', [])
        
        print(f"Initial error: {initial_error:.2f} dB RMS")
        print(f"Final error: {final_error:.2f} dB RMS") 
        print(f"Improvement: {improvement_db:.1f} dB")
        print(f"Total filters: {len(filters)}")
    
    if filters:
        print("\n=== Filter Summary ===")
        for i, filter_info in enumerate(filters):
            filter_type = filter_info.get('filter_type', 'unknown')
            freq = filter_info.get('frequency', 0)
            q = filter_info.get('q', 0)
            gain = filter_info.get('gain_db', 0)
            
            if filter_type == "hp":
                print(f"  Filter {i+1}: High-pass at {freq:.1f}Hz (Q={q:.2f})")
            elif filter_type == "eq":
                print(f"  Filter {i+1}: Peaking EQ at {freq:.1f}Hz, {gain:+.1f}dB (Q={q:.2f})")
            elif filter_type == "ls":
                print(f"  Filter {i+1}: Low shelf at {freq:.1f}Hz, {gain:+.1f}dB (Q={q:.2f})")
            elif filter_type == "hs":
                print(f"  Filter {i+1}: High shelf at {freq:.1f}Hz, {gain:+.1f}dB (Q={q:.2f})")
            else:
                print(f"  Filter {i+1}: {filter_info.get('description', 'Unknown filter')}")

def demo_different_modes(job: Dict[str, Any], rust_binary: str):
    """Demonstrate different output modes."""
    print("=" * 60)
    print("🎯 Demonstrating Different Output Modes")
    print("=" * 60)
    
    # Mode 1: JSON output with progress and result
    print("\n1️⃣  JSON Mode: Progress + Result")
    print("-" * 40)
    steps, result = run_rust_optimizer(job, rust_binary, 
                                     show_progress=True, show_result=True, human_readable=False)
    
    # Mode 2: Human-readable output with progress and result  
    print("\n2️⃣  Human-readable Mode: Progress + Result")
    print("-" * 40)
    steps2, result2 = run_rust_optimizer(job, rust_binary,
                                       show_progress=True, show_result=True, human_readable=True)
    
    # Mode 3: Result only (silent processing)
    print("\n3️⃣  Silent Mode: Result Only")
    print("-" * 40)
    steps3, result3 = run_rust_optimizer(job, rust_binary,
                                       show_progress=False, show_result=True, human_readable=False)
    
    # Mode 4: Progress only (no final result)
    print("\n4️⃣  Monitoring Mode: Progress Only")
    print("-" * 40)
    steps4, result4 = run_rust_optimizer(job, rust_binary,
                                       show_progress=True, show_result=False, human_readable=True)
    
    return steps, result

def main():
    """Main demonstration function."""
    if len(sys.argv) > 1:
        rust_binary = sys.argv[1]
    else:
        rust_binary = "./target/release/roomeq-optimizer"
    
    print("🎵 RoomEQ Rust Optimizer Demo - Command Line Options")
    print("=" * 60)
    
    # Create sample optimization job
    job = create_optimization_job(
        curve_type="weighted_flat",
        preset_type="default", 
        filter_count=6,  # Reduced for faster demo
        sample_rate=48000.0
    )
    
    # Demonstrate different modes
    steps, result = demo_different_modes(job, rust_binary)
    
    # Analyze results
    if steps or result:
        analyze_results(steps, result)
    
    print("\n🎯 Demo completed!")
    print("\n📖 Command Line Usage Examples:")
    print("   # JSON output with progress and result")
    print("   roomeq-optimizer --progress --result < job.json")
    print()
    print("   # Human-readable output")  
    print("   roomeq-optimizer --progress --result --human-readable < job.json")
    print()
    print("   # Silent mode (result only)")
    print("   roomeq-optimizer --result < job.json")
    print()
    print("   # Monitor mode (progress only)")
    print("   roomeq-optimizer --progress --human-readable < job.json")
    print()
    print("   # Completely silent (no output unless error)")
    print("   roomeq-optimizer < job.json")
    
    if not steps and not result:
        print("\n📝 To build and run the Rust optimizer:")
        print("   cd src/rust")
        print("   cargo build --release")
        print("   python3 demo_rust_optimizer.py")

if __name__ == "__main__":
    main()
