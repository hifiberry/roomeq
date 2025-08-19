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

def run_rust_optimizer(job: Dict[str, Any], rust_binary_path: str = "./target/release/roomeq-optimizer") -> List[Dict[str, Any]]:
    """Run the Rust optimizer and collect step outputs."""
    
    # Convert job to JSON
    job_json = json.dumps(job, indent=None, separators=(',', ':'))
    
    print(f"🔧 Starting Rust EQ optimizer...")
    print(f"   Target: {job['target_curve']['name']}")
    print(f"   Preset: {job['optimizer_params']['name']}")
    print(f"   Filters: {job['filter_count']}")
    print(f"   Sample rate: {job['sample_rate']}Hz")
    print()
    
    # Start the Rust process
    try:
        process = subprocess.Popen(
            [rust_binary_path],
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
        
        # Read step outputs from stdout
        for line in process.stdout:
            line = line.strip()
            if line:
                try:
                    step = json.loads(line)
                    steps.append(step)
                    
                    # Print progress
                    print(f"Step {step['step']}: {step['message']}")
                    print(f"  Filters: {len(step['filters'])}")
                    print(f"  Error: {step['residual_error']:.2f} dB RMS")
                    print(f"  Progress: {step['progress_percent']:.1f}%")
                    
                    if step['filters']:
                        last_filter = step['filters'][-1]
                        print(f"  Last filter: {last_filter.get('description', 'Unknown')}")
                    print()
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse step output: {line}")
                    continue
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Read any final messages from stderr
        stderr_output = process.stderr.read()
        if stderr_output:
            print("=== Optimizer Summary ===")
            print(stderr_output)
        
        if return_code != 0:
            print(f"❌ Optimizer failed with return code {return_code}")
            return []
        
        print(f"✅ Optimization completed successfully!")
        return steps
        
    except FileNotFoundError:
        print(f"❌ Rust binary not found: {rust_binary_path}")
        print("   Make sure to compile the Rust code first:")
        print("   cd src/rust && cargo build --release")
        return []
    except Exception as e:
        print(f"❌ Error running optimizer: {e}")
        return []

def analyze_results(steps: List[Dict[str, Any]]):
    """Analyze and summarize optimization results."""
    if not steps:
        print("No optimization steps to analyze.")
        return
    
    print("\n=== Optimization Analysis ===")
    
    first_step = steps[0]
    last_step = steps[-1]
    
    initial_error = first_step['residual_error']
    final_error = last_step['residual_error']
    improvement_db = 20 * np.log10(initial_error / max(final_error, 1e-10))
    
    print(f"Initial error: {initial_error:.2f} dB RMS")
    print(f"Final error: {final_error:.2f} dB RMS") 
    print(f"Improvement: {improvement_db:.1f} dB")
    print(f"Total filters: {len(last_step['filters'])}")
    
    print("\n=== Filter Summary ===")
    for i, filter_info in enumerate(last_step['filters']):
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

def main():
    """Main demonstration function."""
    if len(sys.argv) > 1:
        rust_binary = sys.argv[1]
    else:
        rust_binary = "./target/release/roomeq-optimizer"
    
    print("🎵 RoomEQ Rust Optimizer Demo")
    print("=" * 50)
    
    # Create sample optimization job
    job = create_optimization_job(
        curve_type="weighted_flat",
        preset_type="default", 
        filter_count=8,
        sample_rate=48000.0
    )
    
    # Run optimization
    steps = run_rust_optimizer(job, rust_binary)
    
    # Analyze results
    analyze_results(steps)
    
    print("\n🎯 Demo completed!")
    
    if not steps:
        print("\n📝 To build and run the Rust optimizer:")
        print("   cd src/rust")
        print("   cargo build --release")
        print("   python3 ../../demo_rust_optimizer.py")

if __name__ == "__main__":
    main()
