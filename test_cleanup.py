#!/usr/bin/env python3
"""
Test script for the cleanup functionality.
This script tests the cleanup function without running the full server.
"""

import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import patch

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_cleanup_function():
    """Test the cleanup function in isolation."""
    
    # Import after setting up the path
    from roomeq.roomeq_server import _cleanup_old_recordings, _completed_recordings, _temp_dir
    
    print(f"Testing cleanup functionality...")
    print(f"Temp directory: {_temp_dir}")
    
    # Create some fake recording entries
    # One old recording (20 minutes ago)
    old_recording_id = "old_test_recording"
    old_timestamp = datetime.now() - timedelta(minutes=20)
    old_filename = f"recording_{old_recording_id}.wav"
    old_filepath = os.path.join(_temp_dir, old_filename)
    
    # Create the actual file
    with open(old_filepath, 'w') as f:
        f.write("fake audio data")
    
    _completed_recordings[old_recording_id] = {
        "filename": old_filename,
        "filepath": old_filepath,
        "timestamp": old_timestamp,
        "duration": 5.0,
        "device": "test_device",
        "sample_rate": 48000
    }
    
    # One recent recording (5 minutes ago)
    recent_recording_id = "recent_test_recording"
    recent_timestamp = datetime.now() - timedelta(minutes=5)
    recent_filename = f"recording_{recent_recording_id}.wav"
    recent_filepath = os.path.join(_temp_dir, recent_filename)
    
    # Create the actual file
    with open(recent_filepath, 'w') as f:
        f.write("fake audio data")
    
    _completed_recordings[recent_recording_id] = {
        "filename": recent_filename,
        "filepath": recent_filepath,
        "timestamp": recent_timestamp,
        "duration": 3.0,
        "device": "test_device",
        "sample_rate": 48000
    }
    
    print(f"Created test recordings:")
    print(f"  Old recording: {old_recording_id} ({old_timestamp})")
    print(f"  Recent recording: {recent_recording_id} ({recent_timestamp})")
    print(f"  Total recordings before cleanup: {len(_completed_recordings)}")
    
    # Verify files exist before cleanup
    print(f"  Old file exists: {os.path.exists(old_filepath)}")
    print(f"  Recent file exists: {os.path.exists(recent_filepath)}")
    
    # Run cleanup
    print("\nRunning cleanup...")
    _cleanup_old_recordings()
    
    # Check results
    print(f"  Total recordings after cleanup: {len(_completed_recordings)}")
    print(f"  Old file exists after cleanup: {os.path.exists(old_filepath)}")
    print(f"  Recent file exists after cleanup: {os.path.exists(recent_filepath)}")
    
    # Verify the old recording was removed and recent one remains
    if old_recording_id not in _completed_recordings:
        print("✓ Old recording was removed from memory")
    else:
        print("✗ Old recording was NOT removed from memory")
    
    if recent_recording_id in _completed_recordings:
        print("✓ Recent recording was kept in memory")
    else:
        print("✗ Recent recording was NOT kept in memory")
    
    if not os.path.exists(old_filepath):
        print("✓ Old recording file was deleted")
    else:
        print("✗ Old recording file was NOT deleted")
    
    if os.path.exists(recent_filepath):
        print("✓ Recent recording file was kept")
    else:
        print("✗ Recent recording file was NOT kept")
    
    # Clean up the remaining test file
    try:
        os.unlink(recent_filepath)
        del _completed_recordings[recent_recording_id]
        print("✓ Test cleanup completed")
    except Exception as e:
        print(f"Warning: Could not clean up test files: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_cleanup_function()
