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
import subprocess
import tempfile
import os
import wave
import uuid
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
    
    def _generate_sine_sweep(self, f_start, f_end, duration_samples, amplitude, compensation_mode: str = 'inv_sqrt_f'):
        """
        Generate a logarithmic sine sweep with proper amplitude normalization.
        
        Based on the enDAQ sine sweep testing documentation:
        https://endaq.com/pages/sine-sweep-testing
        
        For logarithmic sweeps, the frequency follows:
        f(t) = f_start * (f_end/f_start)^(t/T)
        
        Compensation modes:
            - 'none': constant amplitude envelope
            - 'inv_sqrt_f': A ∝ 1/sqrt(f) (historical default)
            - 'sqrt_f': A ∝ sqrt(f) (useful for PSD/Hz flatness displays)
        
        Args:
            f_start: Starting frequency in Hz
            f_end: Ending frequency in Hz  
            duration_samples: Duration in samples
            amplitude: Peak amplitude (0.0 to 1.0)
            
        Returns:
            numpy.ndarray: Generated sine sweep samples
        """
        duration_seconds = duration_samples / self.sample_rate
        t = np.linspace(0, duration_seconds, duration_samples, endpoint=False)
        
        # Calculate frequency ratio
        frequency_ratio = f_end / f_start
        
        # Logarithmic frequency sweep formula from enDAQ documentation
        # f(t) = f_start * (f_end/f_start)^(t/T)
        instantaneous_freq = f_start * np.power(frequency_ratio, t / duration_seconds)
        
        # Calculate phase by integrating frequency
        # φ(t) = 2π * ∫ f(τ) dτ from 0 to t
        # For logarithmic sweep: φ(t) = 2π * f_start * T / ln(f_end/f_start) * [(f_end/f_start)^(t/T) - 1]
        log_ratio = np.log(frequency_ratio)
        if abs(log_ratio) > 1e-10:  # Avoid division by zero
            phase = 2 * np.pi * f_start * duration_seconds / log_ratio * (
                np.power(frequency_ratio, t / duration_seconds) - 1
            )
        else:
            # Linear case when f_start ≈ f_end
            phase = 2 * np.pi * f_start * t
        
        # Generate basic sine sweep (unit amplitude)
        signal = np.sin(phase)

        # Amplitude compensation envelope
        comp = np.ones_like(signal)
        mode = (compensation_mode or 'inv_sqrt_f').lower()
        if mode == 'inv_sqrt_f':
            # A ∝ 1/sqrt(f)
            comp = np.sqrt(np.maximum(f_start, 1e-9) / np.maximum(instantaneous_freq, 1e-9))
        elif mode == 'sqrt_f':
            # A ∝ sqrt(f)
            comp = np.sqrt(np.maximum(instantaneous_freq, 1e-9) / np.maximum(f_start, 1e-9))
        elif mode == 'none':
            comp = np.ones_like(signal)
        else:
            logger.warning(f"Unknown compensation_mode '{compensation_mode}', using 'inv_sqrt_f'")
            comp = np.sqrt(np.maximum(f_start, 1e-9) / np.maximum(instantaneous_freq, 1e-9))

        # Normalize envelope to keep peak at 1.0 before applying amplitude
        comp_max = float(np.max(np.abs(comp))) if comp.size else 1.0
        if comp_max > 0:
            comp = comp / comp_max

        # Apply amplitude and compensation
        signal = signal * (amplitude * comp)
        
        # Apply fade-in and fade-out to reduce spectral artifacts
        fade_samples = int(0.01 * self.sample_rate)  # 10ms fade
        if fade_samples > 0:
            # Fade in
            fade_in = np.linspace(0, 1, fade_samples)
            signal[:fade_samples] *= fade_in
            
            # Fade out  
            fade_out = np.linspace(1, 0, fade_samples)
            signal[-fade_samples:] *= fade_out
        
        # Convert to stereo if needed
        if self.channels == 2:
            signal = np.column_stack((signal, signal))
        
        # Convert to S16_LE format
        samples = (signal * 32767).astype(np.int16)
        
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
                       duration: float = 10, amplitude: float = 0.5, repeats: int = 1,
                       compensation_mode: str = 'none'):
        """
        Play sine sweep (logarithmic).
        
        Args:
            start_freq: Starting frequency in Hz (default: 20)
            end_freq: Ending frequency in Hz (default: 20000)
            duration: Duration in seconds (default: 10)
            amplitude: Amplitude scaling (0.0 to 1.0)
            repeats: Number of times to repeat the sweep (default: 1)
            compensation_mode: Amplitude compensation envelope ('none' | 'inv_sqrt_f' | 'sqrt_f')
            
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
            single_sweep = self._generate_sine_sweep(start_freq, end_freq, duration_samples, amplitude, compensation_mode=compensation_mode)
            
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

    def play_sine_sweep_sox(self, start_freq: float = 20, end_freq: float = 20000,
                             duration: float = 10, amplitude: float = 0.5, repeats: int = 1,
                             sox_delay: float = 0.0) -> bool:
        """
        Generate a sine sweep WAV via SoX in /tmp and play it.

        The WAV file is deleted after being loaded into memory.

        Args:
            start_freq: Start frequency in Hz
            end_freq: End frequency in Hz
            duration: Sweep duration in seconds
            amplitude: Linear amplitude scaling (0.0-1.0)
            repeats: Number of times to repeat during playback
            sox_delay: Optional leading silence in seconds added by SoX (default 0.0)

        Returns:
            True if playback started
        """
        if self.playback_thread and self.playback_thread.is_alive():
            logger.warning("Playback already active. Stop current playback first.")
            return False

        # Build temp file path
        tmp_path = os.path.join('/tmp', f"roomeq_sweep_{uuid.uuid4().hex}.wav")
        try:
            # Compose SoX command
            # Example: sox -q -c 2 -n -r 48000 -b 16 /tmp/file.wav synth 10 sine 20/20000 vol 0.5 delay 1
            cmd = [
                'sox', '-q',
                '-c', str(self.channels),
                '-n',
                '-r', str(self.sample_rate),
                '-b', '16',
                tmp_path,
                'synth', f"{duration}", 'sine', f"{start_freq}/{end_freq}"
            ]
            # Apply amplitude via SoX volume effect (linear gain)
            if amplitude is not None and amplitude >= 0.0:
                cmd += ['vol', f"{float(amplitude):.6f}"]
            # Optional delay
            if sox_delay and sox_delay > 0:
                cmd += ['delay', f"{float(sox_delay):.6f}"]

            logger.debug(f"Running SoX command: {' '.join(cmd)}")
            # Run SoX synchronously; raise if it fails
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Verify file exists and has content
            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                logger.error("SoX did not create output file or file is empty")
                return False

            # Load WAV into memory as int16 numpy array
            with wave.open(tmp_path, 'rb') as wf:
                wf_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                nframes = wf.getnframes()

                if sampwidth != 2:
                    logger.warning(f"Expected 16-bit WAV from SoX, got {sampwidth*8}-bit")

                raw = wf.readframes(nframes)
                data = np.frombuffer(raw, dtype=np.int16)

                if wf_channels > 1:
                    data = data.reshape(-1, wf_channels)
                else:
                    # Mono -> optionally duplicate to stereo depending on configured channels
                    if self.channels == 2:
                        data = np.column_stack((data, data))
                    else:
                        # Keep as mono array shape (N,)
                        pass

            # Clean up temp file immediately after loading
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.debug(f"Failed to remove temp WAV {tmp_path}: {e}")

            # Handle repeats by tiling the sample frames
            if repeats and repeats > 1:
                if data.ndim == 2:
                    data = np.tile(data, (repeats, 1))
                else:
                    data = np.tile(data, repeats)

            # Start playback in separate thread
            self.stop_playback = False
            self.playback_thread = threading.Thread(
                target=self._playback_worker,
                args=(data, False),
                daemon=True
            )
            self.playback_thread.start()

            # Optionally wait a brief moment to ensure the thread is running
            time.sleep(0.05)
            return True

        except FileNotFoundError:
            logger.error("SoX is not installed or not found in PATH. Please install 'sox'.")
            # Best effort cleanup
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"SoX command failed: {e.stderr.decode('utf-8', errors='ignore')}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            return False
        except Exception as e:
            logger.error(f"Failed to start SoX-based sweep playback: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            return False


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
    sweep_parser.add_argument('-c', '--compensation-mode', type=str, default='none', choices=['none', 'inv_sqrt_f', 'sqrt_f'],
                            help="Amplitude compensation envelope ('none' | 'inv_sqrt_f' | 'sqrt_f'). Default: 'none'")
    
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
                amplitude=args.amplitude,
                compensation_mode=getattr(args, 'compensation_mode', 'none')
            ):
                generator.wait_for_completion()
        
    except KeyboardInterrupt:
        print("\nInterrupt received, stopping...")
        generator.stop()
    
    return 0


if __name__ == "__main__":
    exit(main())
