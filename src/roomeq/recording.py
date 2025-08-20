#!/usr/bin/env python3
"""
Recording management module for roomeq.
Handles audio recording, file management, and cleanup operations.
"""

import threading
import time
import tempfile
import os
import subprocess
import signal
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from flask import abort

# Configure logging
logger = logging.getLogger(__name__)

# Global recording state and file management
_temp_dir = tempfile.mkdtemp(prefix="roomeq_recordings_")
_active_recordings = {}  # {recording_id: {"thread": thread, "filename": filename, "status": status}}
_completed_recordings = {}  # {recording_id: {"filename": filename, "timestamp": datetime, "duration": float}}
_cleanup_timer = None


class RecordingManager:
    """Manages audio recordings, file storage, and cleanup operations."""
    
    def __init__(self):
        self.temp_dir = _temp_dir
        self.active_recordings = _active_recordings
        self.completed_recordings = _completed_recordings
        self.cleanup_timer = None
    
    def get_temp_dir(self) -> str:
        """Get the temporary directory for recordings."""
        return self.temp_dir
    
    def get_active_recordings(self) -> Dict[str, Any]:
        """Get active recordings dictionary."""
        return self.active_recordings
    
    def get_completed_recordings(self) -> Dict[str, Any]:
        """Get completed recordings dictionary."""
        return self.completed_recordings
    
    def start_cleanup_timer(self):
        """Start the cleanup timer."""
        start_cleanup_timer()
    
    def stop_cleanup_timer(self):
        """Stop the cleanup timer."""
        stop_cleanup_timer()
    
    def cleanup_old_recordings(self):
        """Run cleanup manually."""
        cleanup_old_recordings()
    
    def validate_recording_file(self, filename: str) -> str:
        """Validate recording file."""
        return validate_recording_file(filename)
    
    def start_recording(self, recording_id: str, device: str, duration: float, sample_rate: int = 48000):
        """Start a recording."""
        return start_recording(recording_id, device, duration, sample_rate)
    
    def register_generated_file(self, file_id: str, filepath: str, file_type: str = "generated", metadata: dict = None):
        """Register a generated file (sweep/noise) for automatic cleanup."""
        filename = os.path.basename(filepath)
        
        # Add to completed recordings with generated file metadata
        self.completed_recordings[file_id] = {
            "filename": filename,
            "filepath": filepath, 
            "timestamp": datetime.now(),
            "duration": metadata.get("duration", 0) if metadata else 0,
            "device": metadata.get("device", "generated") if metadata else "generated",
            "sample_rate": metadata.get("sample_rate", 48000) if metadata else 48000,
            "file_type": file_type,  # Mark as generated file
            "signal_type": metadata.get("signal_type") if metadata else None,
            "parameters": metadata.get("parameters", {}) if metadata else {}
        }
        
        logger.info(f"Registered generated file for cleanup: {filename} (ID: {file_id})")
        return file_id


def recording_worker(recording_id: str, device: str, duration: float, sample_rate: int = 48000):
    """Background worker for audio recording."""
    global _active_recordings, _completed_recordings
    
    filename = f"recording_{recording_id}.wav"
    filepath = os.path.join(_temp_dir, filename)
    
    try:
        logger.info(f"Starting recording {recording_id}: {duration}s on device {device}")
        
        # Update status to recording
        _active_recordings[recording_id]["status"] = "recording"
        
        # Use arecord to capture audio
        cmd = [
            'arecord',
            '-D', device,
            '-f', 'S16_LE',
            '-c', '1',  # Mono recording
            '-r', str(sample_rate),
            '-d', str(int(duration)) if duration == int(duration) else str(duration),
            filepath
        ]
        
        start_time = datetime.now()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 10)
        end_time = datetime.now()
        
        if result.returncode == 0:
            # Recording successful
            actual_duration = (end_time - start_time).total_seconds()
            
            # Move to completed recordings
            _completed_recordings[recording_id] = {
                "filename": filename,
                "filepath": filepath,
                "timestamp": start_time,
                "duration": actual_duration,
                "device": device,
                "sample_rate": sample_rate
            }
            
            logger.info(f"Recording {recording_id} completed successfully: {actual_duration:.1f}s")
            
        else:
            logger.error(f"Recording {recording_id} failed: {result.stderr}")
            # Clean up failed recording file
            try:
                os.unlink(filepath)
            except:
                pass
                
    except subprocess.TimeoutExpired:
        logger.error(f"Recording {recording_id} timed out")
        try:
            os.unlink(filepath)
        except:
            pass
    except Exception as e:
        logger.error(f"Recording {recording_id} error: {e}")
        try:
            os.unlink(filepath)
        except:
            pass
    finally:
        # Remove from active recordings
        if recording_id in _active_recordings:
            del _active_recordings[recording_id]


def validate_recording_file(filename: str) -> str:
    """Validate and sanitize recording filename for security."""
    # Only allow wav files
    if not filename.endswith('.wav'):
        abort(400, "Only WAV files are allowed")
    
    # Remove any path components for security
    filename = os.path.basename(filename)
    
    # Check if file exists in temp directory
    filepath = os.path.join(_temp_dir, filename)
    if not os.path.exists(filepath):
        abort(404, "Recording file not found")
    
    # Ensure file is actually in our temp directory (prevent directory traversal)
    try:
        real_temp = os.path.realpath(_temp_dir)
        real_file = os.path.realpath(filepath)
        if not real_file.startswith(real_temp):
            abort(403, "Access denied")
    except:
        abort(403, "Access denied")
    
    return filepath


def cleanup_old_recordings():
    """Clean up recordings older than 10 minutes."""
    global _completed_recordings
    
    cutoff_time = datetime.now() - timedelta(minutes=10)
    recordings_to_remove = []
    
    logger.info("Running cleanup job for old recordings")
    
    for recording_id, recording in _completed_recordings.items():
        if recording["timestamp"] < cutoff_time:
            recordings_to_remove.append(recording_id)
    
    for recording_id in recordings_to_remove:
        try:
            recording = _completed_recordings[recording_id]
            filepath = recording["filepath"]
            
            # Remove the file
            if os.path.exists(filepath):
                os.unlink(filepath)
                logger.info(f"Cleaned up old recording file: {recording['filename']}")
            
            # Remove from completed recordings
            del _completed_recordings[recording_id]
            logger.info(f"Removed old recording from memory: {recording_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up recording {recording_id}: {e}")
    
    if recordings_to_remove:
        logger.info(f"Cleanup job completed: removed {len(recordings_to_remove)} old recordings")
    else:
        logger.debug("Cleanup job completed: no old recordings to remove")


def start_cleanup_timer():
    """Start the periodic cleanup timer (runs every 60 minutes)."""
    global _cleanup_timer
    
    def run_cleanup():
        cleanup_old_recordings()
        # Schedule next cleanup
        start_cleanup_timer()
    
    _cleanup_timer = threading.Timer(3600.0, run_cleanup)  # 3600 seconds = 60 minutes
    _cleanup_timer.daemon = True
    _cleanup_timer.start()
    logger.info("Started cleanup timer: will run every 60 minutes")


def stop_cleanup_timer():
    """Stop the cleanup timer."""
    global _cleanup_timer
    if _cleanup_timer and _cleanup_timer.is_alive():
        _cleanup_timer.cancel()
        logger.info("Stopped cleanup timer")


def start_recording(recording_id: str, device: str, duration: float, sample_rate: int = 48000):
    """Start a new recording in a background thread."""
    global _active_recordings
    
    # Create thread for recording
    thread = threading.Thread(
        target=recording_worker,
        args=(recording_id, device, duration, sample_rate)
    )
    thread.daemon = True
    
    # Add to active recordings
    _active_recordings[recording_id] = {
        "thread": thread,
        "filename": f"recording_{recording_id}.wav",
        "status": "starting",
        "device": device,
        "duration": duration,
        "sample_rate": sample_rate,
        "start_time": datetime.now()
    }
    
    # Start the recording thread
    thread.start()
    
    return recording_id


def get_recording_status(recording_id: str) -> Optional[Dict[str, Any]]:
    """Get the status of a recording."""
    global _active_recordings, _completed_recordings
    
    if recording_id in _active_recordings:
        recording = _active_recordings[recording_id]
        return {
            "status": "active",
            "recording_status": recording["status"],
            "device": recording["device"],
            "duration": recording["duration"],
            "sample_rate": recording["sample_rate"],
            "start_time": recording["start_time"].isoformat(),
            "filename": recording["filename"]
        }
    
    if recording_id in _completed_recordings:
        recording = _completed_recordings[recording_id]
        return {
            "status": "completed",
            "filename": recording["filename"],
            "filepath": recording["filepath"],
            "timestamp": recording["timestamp"].isoformat(),
            "duration": recording["duration"],
            "device": recording["device"],
            "sample_rate": recording["sample_rate"]
        }
    
    return None


def list_recordings() -> Dict[str, Any]:
    """List all recordings (active and completed)."""
    global _active_recordings, _completed_recordings
    
    active_list = []
    for recording_id, recording in _active_recordings.items():
        active_list.append({
            "recording_id": recording_id,
            "status": recording["status"],
            "device": recording["device"],
            "duration": recording["duration"],
            "sample_rate": recording["sample_rate"],
            "start_time": recording["start_time"].isoformat(),
            "filename": recording["filename"]
        })
    
    completed_list = []
    for recording_id, recording in _completed_recordings.items():
        completed_list.append({
            "recording_id": recording_id,
            "filename": recording["filename"],
            "timestamp": recording["timestamp"].isoformat(),
            "duration": recording["duration"],
            "device": recording["device"],
            "sample_rate": recording["sample_rate"]
        })
    
    return {
        "active_recordings": active_list,
        "completed_recordings": completed_list,
        "active_count": len(active_list),
        "completed_count": len(completed_list)
    }


def delete_recording(recording_id: str) -> bool:
    """Delete a recording by ID."""
    global _completed_recordings
    
    if recording_id not in _completed_recordings:
        return False
    
    recording = _completed_recordings[recording_id]
    filepath = recording["filepath"]
    
    try:
        # Remove the file
        if os.path.exists(filepath):
            os.unlink(filepath)
            logger.info(f"Deleted recording file: {recording['filename']}")
        
        # Remove from completed recordings
        del _completed_recordings[recording_id]
        logger.info(f"Removed recording from memory: {recording_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error deleting recording {recording_id}: {e}")
        return False


def delete_recording_file(filename: str) -> bool:
    """Delete a recording file by filename."""
    global _completed_recordings
    
    try:
        filepath = validate_recording_file(filename)
        
        os.unlink(filepath)
        logger.info(f"Deleted recording file: {filename}")
        
        # Also remove from completed recordings if present
        recording_id_to_remove = None
        for recording_id, recording in _completed_recordings.items():
            if recording["filename"] == filename:
                recording_id_to_remove = recording_id
                break
        
        if recording_id_to_remove:
            del _completed_recordings[recording_id_to_remove]
        
        return True
        
    except Exception as e:
        logger.error(f"Error deleting recording file {filename}: {e}")
        return False


def get_cleanup_status() -> Dict[str, Any]:
    """Get current cleanup status and settings."""
    global _cleanup_timer, _completed_recordings
    
    cutoff_time = datetime.now() - timedelta(minutes=10)
    old_recordings = []
    
    for recording_id, recording in _completed_recordings.items():
        if recording["timestamp"] < cutoff_time:
            age_minutes = (datetime.now() - recording["timestamp"]).total_seconds() / 60
            old_recordings.append({
                "recording_id": recording_id,
                "filename": recording["filename"],
                "timestamp": recording["timestamp"].isoformat(),
                "age_minutes": round(age_minutes, 1)
            })
    
    return {
        "cleanup_interval_minutes": 60,
        "max_age_minutes": 10,
        "timer_active": _cleanup_timer is not None and _cleanup_timer.is_alive(),
        "total_recordings": len(_completed_recordings),
        "old_recordings_count": len(old_recordings),
        "old_recordings": old_recordings
    }


def signal_handler(signum, frame):
    """Handle shutdown signals to cleanup resources."""
    logger.info(f"Received signal {signum}, shutting down...")
    stop_cleanup_timer()
    sys.exit(0)


# Create a global instance for easy access
recording_manager = RecordingManager()
