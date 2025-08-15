#!/usr/bin/env python3
"""
Signal generator module for RoomEQ.

Generates audio signals using ALSA output for acoustic measurements.
"""

import time
import threading
import signal
import sys
import logging
from typing import Optional
import numpy as np
import alsaaudio

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Audio signal generator using ALSA output."""
    
    def __init__(self, device: Optional[str] = None, sample_rate: int = 48000, channels: int = 2):
        """
        Initialize the signal generator.
        
        Args:
            device: ALSA output device name (e.g., "hw:0,0", "default"). If None, uses "default".
            sample_rate: Sample rate in Hz (default: 48000)
            channels: Number of output channels (default: 2 for stereo)
        """
        self.device = device or "default"
        self.sample_rate = sample_rate
        self.channels = channels
        self.format_type = alsaaudio.PCM_FORMAT_S16_LE
        self.pcm = None
        self.stop_playback = False
        self.playback_thread = None
        
        logger.debug(f"SignalGenerator: device={self.device}, rate={sample_rate}, channels={channels}")
    
    def _open_playback(self):
        """Open ALSA PCM device for playback."""
        try:
            # For alsaaudio 0.8, use positional parameters and set properties after creation
            self.pcm = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK, alsaaudio.PCM_NORMAL, self.device)
            self.pcm.setchannels(self.channels)
            self.pcm.setrate(self.sample_rate)
            self.pcm.setformat(self.format_type)
            self.pcm.setperiodsize(1024)
            logger.debug(f"Opened playback device: {self.device}")
            
        except alsaaudio.ALSAAudioError as e:
            raise RuntimeError(f"Failed to open ALSA playback device {self.device}: {e}")
    
    def _close_playback(self):
        """Close ALSA PCM device."""
        if self.pcm:
            self.pcm.close()
            self.pcm = None
    
    def _generate_noise(self, duration_samples: int, amplitude: float = 0.5) -> np.ndarray:
        """
        Generate white noise samples.
        
        Args:
            duration_samples: Number of samples to generate
            amplitude: Amplitude scaling (0.0 to 1.0)
            
        Returns:
            Noise samples as numpy array
        """
        # Generate white noise
        noise = np.random.uniform(-1.0, 1.0, duration_samples) * amplitude
        
        # Convert to stereo if needed
        if self.channels == 2:
            noise = np.column_stack((noise, noise))
        
        # Convert to S16_LE format
        samples = (noise * 32767).astype(np.int16)
        
        return samples
    
    def _generate_sine_sweep(self, start_freq: float, end_freq: float, 
                           duration_samples: int, amplitude: float = 0.5) -> np.ndarray:
        """
        Generate sine sweep (chirp) samples.
        
        Args:
            start_freq: Starting frequency in Hz
            end_freq: Ending frequency in Hz
            duration_samples: Number of samples to generate
            amplitude: Amplitude scaling (0.0 to 1.0)
            
        Returns:
            Sweep samples as numpy array
        """
        t = np.linspace(0, duration_samples / self.sample_rate, duration_samples, False)
        
        # Generate logarithmic sweep with proper formula for constant power
        # The correct logarithmic chirp formula maintains constant power across frequencies
        # f(t) = f0 * (f1/f0)^(t/T)
        # φ(t) = 2π * f0 * T/ln(f1/f0) * ((f1/f0)^(t/T) - 1)
        
        duration_sec = duration_samples / self.sample_rate
        frequency_ratio = end_freq / start_freq
        log_ratio = np.log(frequency_ratio)
        
        # Calculate instantaneous phase for logarithmic sweep
        phase = 2 * np.pi * start_freq * duration_sec / log_ratio * (frequency_ratio ** (t / duration_sec) - 1)
        
        # Generate the sweep with constant amplitude
        sweep = np.sin(phase) * amplitude
        
        # Convert to stereo if needed
        if self.channels == 2:
            sweep = np.column_stack((sweep, sweep))
        
        # Convert to S16_LE format
        samples = (sweep * 32767).astype(np.int16)
        
        return samples
    
    def _playback_worker(self, samples: np.ndarray, loop: bool = False, continuous_noise: bool = False, amplitude: float = 0.5):
        """
        Worker function for audio playback in a separate thread.
        
        Args:
            samples: Audio samples to play
            loop: Whether to loop the audio
            continuous_noise: Whether to generate continuous new noise samples
            amplitude: Amplitude for continuous noise generation
        """
        try:
            self._open_playback()
            
            # Calculate chunk size for smooth playback
            chunk_size = 1024 * self.channels
            
            if continuous_noise:
                # For continuous noise, generate new samples continuously
                while not self.stop_playback:
                    # Generate a small chunk of new noise samples
                    noise_chunk = self._generate_noise(chunk_size // self.channels if self.channels == 2 else chunk_size, amplitude)
                    data = noise_chunk.tobytes()
                    self.pcm.write(data)
            else:
                # Original behavior for non-noise signals
                while not self.stop_playback:
                    for i in range(0, len(samples), chunk_size):
                        if self.stop_playback:
                            break
                            
                        chunk = samples[i:i + chunk_size]
                        
                        # Pad last chunk if needed
                        if len(chunk) < chunk_size:
                            if self.channels == 2:
                                padding_shape = (chunk_size - len(chunk), 2)
                            else:
                                padding_shape = (chunk_size - len(chunk),)
                            padding = np.zeros(padding_shape, dtype=np.int16)
                            chunk = np.vstack((chunk, padding)) if self.channels == 2 else np.concatenate((chunk, padding))
                        
                        # Convert to bytes and write
                        data = chunk.tobytes()
                        self.pcm.write(data)
                    
                    if not loop:
                        break
                        
        except Exception as e:
            logger.error(f"Playback error: {e}")
        finally:
            self._close_playback()
    
    def play_noise(self, duration: float = 0, amplitude: float = 0.5):
        """
        Play white noise.
        
        Args:
            duration: Duration in seconds. If 0, plays indefinitely until stopped.
            amplitude: Amplitude scaling (0.0 to 1.0)
            
        Returns:
            True if started successfully
        """
        if self.playback_thread and self.playback_thread.is_alive():
            logger.warning("Playback already active. Stop current playback first.")
            return False
        
        try:
            if duration == 0:
                # Infinite playback - use continuous noise generation
                print(f"Playing white noise indefinitely at {amplitude:.1%} amplitude...")
                print("Press Ctrl+C to stop.")
                
                # Start playback with continuous noise generation
                self.stop_playback = False
                self.playback_thread = threading.Thread(
                    target=self._playback_worker, 
                    args=(np.array([]), False, True, amplitude),  # empty samples, no loop, continuous noise, amplitude
                    daemon=True
                )
                self.playback_thread.start()
            else:
                duration_samples = int(duration * self.sample_rate)
                print(f"Playing white noise for {duration} seconds at {amplitude:.1%} amplitude...")
                
                # Generate noise samples for fixed duration
                samples = self._generate_noise(duration_samples, amplitude)
                
                # Start playback in separate thread
                self.stop_playback = False
                self.playback_thread = threading.Thread(
                    target=self._playback_worker, 
                    args=(samples, False, False, amplitude),  # samples, no loop, no continuous noise
                    daemon=True
                )
                self.playback_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start noise playback: {e}")
            return False
    
    def play_sine_sweep(self, start_freq: float = 20, end_freq: float = 20000, 
                       duration: float = 10, amplitude: float = 0.5, repeats: int = 1):
        """
        Play sine sweep (logarithmic).
        
        Args:
            start_freq: Starting frequency in Hz (default: 20)
            end_freq: Ending frequency in Hz (default: 20000)
            duration: Duration in seconds (default: 10)
            amplitude: Amplitude scaling (0.0 to 1.0)
            repeats: Number of times to repeat the sweep (default: 1)
            
        Returns:
            True if started successfully
        """
        if self.playback_thread and self.playback_thread.is_alive():
            logger.warning("Playback already active. Stop current playback first.")
            return False
        
        try:
            duration_samples = int(duration * self.sample_rate)
            
            if repeats == 1:
                print(f"Playing sine sweep: {start_freq} Hz → {end_freq} Hz over {duration} seconds at {amplitude:.1%} amplitude...")
            else:
                total_duration = duration * repeats
                print(f"Playing {repeats} sine sweeps: {start_freq} Hz → {end_freq} Hz, {duration}s each (total: {total_duration}s) at {amplitude:.1%} amplitude...")
            
            # Generate single sweep samples
            single_sweep = self._generate_sine_sweep(start_freq, end_freq, duration_samples, amplitude)
            
            # If multiple repeats, concatenate the sweep samples
            if repeats > 1:
                # Create array for all sweeps
                if len(single_sweep.shape) == 1:
                    # Mono
                    samples = np.tile(single_sweep, repeats)
                else:
                    # Stereo
                    samples = np.tile(single_sweep, (repeats, 1))
            else:
                samples = single_sweep
            
            # Start playback in separate thread
            self.stop_playback = False
            self.playback_thread = threading.Thread(
                target=self._playback_worker, 
                args=(samples, False), 
                daemon=True
            )
            self.playback_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start sine sweep playback: {e}")
            return False
    
    def stop(self):
        """Stop current playback."""
        if self.playback_thread and self.playback_thread.is_alive():
            print("\nStopping playback...")
            self.stop_playback = True
            self.playback_thread.join(timeout=2.0)
            return True
        return False
    
    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self.playback_thread is not None and self.playback_thread.is_alive()
    
    def wait_for_completion(self):
        """Wait for current playback to complete."""
        if self.playback_thread:
            self.playback_thread.join()


def main():
    """Command-line interface for signal generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio Signal Generator')
    subparsers = parser.add_subparsers(dest='command', help='Signal type')
    
    # Noise command
    noise_parser = subparsers.add_parser('noise', help='Generate white noise')
    noise_parser.add_argument('-d', '--device', type=str, default=None,
                            help='ALSA output device (default: "default")')
    noise_parser.add_argument('-t', '--time', type=float, default=0,
                            help='Duration in seconds (0 = infinite, default: 0)')
    noise_parser.add_argument('-a', '--amplitude', type=float, default=0.5,
                            help='Amplitude 0.0-1.0 (default: 0.5)')
    
    # Sweep command
    sweep_parser = subparsers.add_parser('sweep', help='Generate sine sweep')
    sweep_parser.add_argument('-d', '--device', type=str, default=None,
                            help='ALSA output device (default: "default")')
    sweep_parser.add_argument('-s', '--start', type=float, default=20,
                            help='Start frequency in Hz (default: 20)')
    sweep_parser.add_argument('-e', '--end', type=float, default=20000,
                            help='End frequency in Hz (default: 20000)')
    sweep_parser.add_argument('-t', '--time', type=float, default=10,
                            help='Duration in seconds (default: 10)')
    sweep_parser.add_argument('-a', '--amplitude', type=float, default=0.5,
                            help='Amplitude 0.0-1.0 (default: 0.5)')
    
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Set up logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s:%(name)s:%(message)s')
    
    # Create signal generator
    generator = SignalGenerator(device=args.device)
    
    # Set up signal handler for Ctrl+C
    def signal_handler(sig, frame):
        print("\nInterrupt received, stopping...")
        generator.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if args.command == 'noise':
            if generator.play_noise(duration=args.time, amplitude=args.amplitude):
                generator.wait_for_completion()
        
        elif args.command == 'sweep':
            if generator.play_sine_sweep(
                start_freq=args.start, 
                end_freq=args.end, 
                duration=args.time, 
                amplitude=args.amplitude
            ):
                generator.wait_for_completion()
        
    except KeyboardInterrupt:
        print("\nInterrupt received, stopping...")
        generator.stop()
    
    return 0


if __name__ == "__main__":
    exit(main())
