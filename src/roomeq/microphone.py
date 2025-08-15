#!/usr/bin/env python3
"""
Microphone detection module for RoomEQ.

This module identifies connected microphones and their sensitivity settings
using the native Python ALSA API instead of reading from /proc.
"""

import alsaaudio
import subprocess
import re
import logging
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class MicrophoneDetector:
    """Detects and identifies connected microphones with their properties."""
    
    # Known microphone configurations: (usb_id, name) -> (device_name, sensitivity)
    KNOWN_MICROPHONES = {
        ("0d8c:0134", "Microphone"): ("HiFiBerry Mic", "115.5"),
        (None, "UMM6"): ("Dayton UMM6", "137.5"),
        (None, "U18dB"): ("MiniDSP Umik", "115"),
    }
    
    def __init__(self):
        """Initialize the microphone detector."""
        self.audio_cards = self._get_audio_cards()
    
    def _get_audio_cards(self) -> List[str]:
        """Get list of available audio cards from ALSA."""
        try:
            return alsaaudio.cards()
        except Exception as e:
            logger.error(f"Failed to get audio cards: {e}")
            return []
    
    def _get_card_index(self, card_name: str) -> Optional[int]:
        """Get the card index for a given card name."""
        try:
            # ALSA card names are typically in format 'hw:CARD=name,DEV=0'
            # We need to find the actual card index
            cards = alsaaudio.cards()
            for i, name in enumerate(cards):
                if name == card_name:
                    return i
            return None
        except Exception as e:
            logger.error(f"Failed to get card index for {card_name}: {e}")
            return None
    
    def _has_input_capability(self, card_name: str) -> bool:
        """Check if a card has input (capture) capability."""
        try:
            card_index = self._get_card_index(card_name)
            if card_index is None:
                return False
            
            # Try to open a PCM device for capture on this card
            try:
                # For alsaaudio 0.8, use positional parameters
                pcm = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, f"hw:{card_index}")
                pcm.close()
                return True
            except alsaaudio.ALSAAudioError:
                # If we can't open for capture, it doesn't have input capability
                return False
                
        except Exception as e:
            logger.debug(f"Error checking input capability for {card_name}: {e}")
            return False
    
    def _get_usb_id_from_proc(self, card_index: int) -> Optional[str]:
        """Get USB ID from /proc as fallback when ALSA doesn't provide it."""
        try:
            with open(f"/proc/asound/card{card_index}/usbid", "r") as f:
                return f.read().strip()
        except (FileNotFoundError, IOError):
            return None
    
    def _get_card_name_from_proc(self, card_index: int) -> Optional[str]:
        """Get card name from /proc as fallback."""
        try:
            with open(f"/proc/asound/card{card_index}/id", "r") as f:
                return f.read().strip()
        except (FileNotFoundError, IOError):
            return None
    
    def _get_usb_device_info(self, card_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get USB device information using ALSA and fallback to /proc.
        Returns (usb_id, device_name) tuple.
        """
        usb_id = None
        device_name = None
        
        # Try to get card index
        card_index = self._get_card_index(card_name)
        
        # Get device name from ALSA card info
        try:
            # ALSA card names are the device names in most cases
            device_name = card_name
        except Exception as e:
            logger.debug(f"Could not get device name from ALSA for {card_name}: {e}")
        
        # Try to get USB ID and name from /proc as fallback
        if card_index is not None:
            proc_usb_id = self._get_usb_id_from_proc(card_index)
            proc_name = self._get_card_name_from_proc(card_index)
            
            if proc_usb_id:
                usb_id = proc_usb_id
            if proc_name:
                device_name = proc_name
        
        return usb_id, device_name
    
    def _identify_microphone(self, usb_id: Optional[str], device_name: Optional[str]) -> Tuple[str, str]:
        """
        Identify microphone based on USB ID and device name.
        Returns (identified_device, sensitivity) tuple.
        """
        if not device_name:
            return "", "0"
        
        # Check known microphones
        for (known_usb_id, known_name), (device, sensitivity) in self.KNOWN_MICROPHONES.items():
            if known_usb_id and usb_id:
                # Match by USB ID and name
                if usb_id == known_usb_id and device_name == known_name:
                    return device, sensitivity
            else:
                # Match by name only
                if device_name == known_name:
                    return device, sensitivity
        
        # Unknown device
        return f"Unknown ({device_name})", "0"
    
    def _get_microphone_gain(self, card_index: int) -> Optional[float]:
        """
        Get the current microphone gain for a specific card.
        
        Args:
            card_index: ALSA card index
            
        Returns:
            Current gain in dB or None if not available
        """
        try:
            # Try to get mixer controls for the card
            result = subprocess.run(
                ['amixer', '-c', str(card_index), 'sget', 'Mic'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                # Parse the output to extract gain value
                # Look for pattern like "Capture 20 [100%] [20.00dB] [on]"
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Capture' in line and 'dB' in line:
                        # Extract dB value using regex
                        db_match = re.search(r'\[([+-]?\d+(?:\.\d+)?)dB\]', line)
                        if db_match:
                            gain_db = float(db_match.group(1))
                            logger.debug(f"Found microphone gain for card {card_index}: {gain_db} dB")
                            return gain_db
                            
            # If Mic control doesn't exist, try other common control names
            for control_name in ['Microphone', 'Capture', 'Input']:
                result = subprocess.run(
                    ['amixer', '-c', str(card_index), 'sget', control_name],
                    capture_output=True, text=True, timeout=5
                )
                
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'Capture' in line and 'dB' in line:
                            db_match = re.search(r'\[([+-]?\d+(?:\.\d+)?)dB\]', line)
                            if db_match:
                                gain_db = float(db_match.group(1))
                                logger.debug(f"Found {control_name} gain for card {card_index}: {gain_db} dB")
                                return gain_db
                                
        except Exception as e:
            logger.debug(f"Failed to get gain for card {card_index}: {e}")
            
        return None
    
    def detect_microphones(self) -> List[Tuple[int, str, str, Optional[float]]]:
        """
        Detect all connected microphones.
        Returns list of (card_index, device_name, sensitivity, gain_db) tuples.
        """
        microphones = []
        
        for card_name in self.audio_cards:
            # Check if card has input capability
            if not self._has_input_capability(card_name):
                continue
            
            card_index = self._get_card_index(card_name)
            if card_index is None:
                continue
            
            # Get USB device information
            usb_id, device_name = self._get_usb_device_info(card_name)
            
            # Identify the microphone
            identified_device, sensitivity = self._identify_microphone(usb_id, device_name)
            
            # Get microphone gain
            gain_db = self._get_microphone_gain(card_index)
            
            if identified_device:
                microphones.append((card_index, identified_device, sensitivity, gain_db))
        
        return microphones
    
    def get_audio_inputs(self) -> List[int]:
        """
        Get audio input card indices, similar to /opt/hifiberry/bin/audio-inputs.
        This is a simplified version that returns cards with input capability.
        """
        input_cards = []
        
        for card_name in self.audio_cards:
            if self._has_input_capability(card_name):
                card_index = self._get_card_index(card_name)
                if card_index is not None:
                    input_cards.append(card_index)
        
        return input_cards


def detect_microphones() -> List[str]:
    """
    Detect microphones and return formatted output similar to the bash script.
    Returns list of strings in format: "card_index:device_name:sensitivity:gain"
    """
    detector = MicrophoneDetector()
    microphones = detector.detect_microphones()
    
    result = []
    for card_index, device_name, sensitivity, gain in microphones:
        gain_str = f"{gain}" if gain is not None else "N/A"
        result.append(f"{card_index}:{device_name}:{sensitivity}:{gain_str}")
    
    return result


def main():
    """Main function for command-line usage."""
    try:
        microphones = detect_microphones()
        for mic in microphones:
            print(mic)
    except Exception as e:
        logger.error(f"Error detecting microphones: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
