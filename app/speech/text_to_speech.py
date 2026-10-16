"""Text-to-Speech (TTS) module for SignBridge.

Provides asynchronous and synchronous speech synthesis across platforms:
- Windows: Native Microsoft SAPI5 via win32com with background worker thread,
  supporting continuous repeated utterances without dropped speech
- macOS: Native 'say' command with rate and voice customization
- Cross-platform fallbacks: pyttsx3 and PowerShell SAPI

Includes cancellation of rapid calls to prevent audio clash and non-overlapping playback.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Text-to-speech engine supporting Windows, macOS, and Linux.

    Runs speech asynchronously in a background worker thread.
    Rapid calls do not overlap — new speech requests cancel any ongoing speech
    and pending requests so only the latest request plays.

    Args:
        rate: Speech rate in words per minute (default: 180).
        voice: Voice name or identifier.
    """

    DEFAULT_RATE: int = 180

    def __init__(
        self,
        rate: int = DEFAULT_RATE,
        voice: Optional[str] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._rate: int = int(rate) if rate and rate > 0 else self.DEFAULT_RATE
        self._voice: Optional[str] = voice.strip() if voice and voice.strip() else None

        self._process: Optional[subprocess.Popen] = None
        self._speaking: bool = False
        self._say_binary: Optional[str] = shutil.which("say")

        # Background worker for Windows SAPI / cross-platform
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown: bool = False

        if self._say_binary:
            logger.info("Text-to-Speech: Using macOS 'say' binary at %s", self._say_binary)
        else:
            self._start_worker()

    def _start_worker(self) -> None:
        """Start background worker thread for speech synthesis."""
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="SignBridge-TTS-Worker",
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Worker loop that handles speech requests reliably without dropping utterances."""
        # Initialize COM on Windows for this thread
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            try:
                import comtypes
                comtypes.CoInitialize()
            except Exception:
                pass

        sapi_voice = None
        if sys.platform == "win32":
            try:
                import win32com.client
                sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
                logger.info("Text-to-Speech: Native SAPI.SpVoice initialized successfully")
            except Exception as e:
                logger.warning("win32com SAPI init failed: %s", e)

        pyttsx_engine = None
        if sapi_voice is None:
            try:
                import pyttsx3
                pyttsx_engine = pyttsx3.init()
                logger.info("Text-to-Speech: pyttsx3 engine initialized as fallback")
            except Exception as e:
                logger.warning("pyttsx3 initialization failed: %s", e)

        while not self._shutdown:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None or self._shutdown:
                break

            text, rate, voice, done_event = item
            self._stop_event.clear()

            with self._lock:
                self._speaking = True

            try:
                if sapi_voice is not None:
                    # Map rate in WPM (180 -> 0) to SAPI (-10 to 10)
                    if rate:
                        sapi_rate = max(-10, min(10, int((rate - 180) / 15)))
                        try:
                            sapi_voice.Rate = sapi_rate
                        except Exception:
                            pass

                    # Select voice if specified
                    if voice:
                        try:
                            tokens = sapi_voice.GetVoices()
                            for i in range(tokens.Count):
                                tok = tokens.Item(i)
                                if voice.lower() in tok.GetDescription().lower():
                                    sapi_voice.Voice = tok
                                    break
                        except Exception:
                            pass

                    # SVSFlagsAsync = 1, SVSFPurgeBeforeSpeak = 2
                    sapi_voice.Speak(text, 1 | 2)

                    # Poll until finished speaking or stopped
                    while not sapi_voice.WaitUntilDone(30):
                        if self._stop_event.is_set() or self._shutdown:
                            sapi_voice.Speak("", 2)  # Purge immediately
                            break

                elif pyttsx_engine is not None:
                    if rate:
                        try:
                            pyttsx_engine.setProperty("rate", rate)
                        except Exception:
                            pass
                    pyttsx_engine.say(text)
                    pyttsx_engine.runAndWait()

                elif sys.platform == "win32":
                    self._speak_powershell(text)
                else:
                    logger.info("TTS Simulation: %s", text)

            except Exception as e:
                logger.error("Error during speech synthesis: %s", e)
                if sys.platform == "win32" and sapi_voice is None:
                    try:
                        self._speak_powershell(text)
                    except Exception:
                        pass
            finally:
                with self._lock:
                    self._speaking = False
                if done_event is not None:
                    done_event.set()
                self._queue.task_done()

    def _speak_powershell(self, text: str) -> None:
        """Windows PowerShell SAPI fallback."""
        escaped = text.replace("'", "''").replace('"', '`"')
        cmd = [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Speak('{escaped}')",
        ]
        with self._lock:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        try:
            self._process.wait()
        except Exception:
            pass
        finally:
            with self._lock:
                self._process = None

    def speak(self, text: str) -> None:
        """Speak text asynchronously in a background thread.

        Cancels any ongoing speech before starting new speech so rapid
        calls do not overlap and only the latest request plays.

        Args:
            text: The text to speak aloud.
        """
        if not text or not isinstance(text, str):
            return

        clean_text = text.strip()
        if not clean_text:
            return

        # Cancel any ongoing speech so rapid calls do not overlap
        self.stop()

        thread = threading.Thread(
            target=self._speak_impl,
            args=(clean_text,),
            daemon=True,
            name="SignBridge-TTS-Async",
        )
        thread.start()

    def speak_sync(self, text: str) -> None:
        """Speak text synchronously (blocks until playback finishes).

        Args:
            text: The text to speak aloud.
        """
        if not text or not isinstance(text, str):
            return

        clean_text = text.strip()
        if not clean_text:
            return

        self.stop()
        self._speak_impl(clean_text)

    def _speak_impl(self, text: str) -> None:
        """Internal implementation that runs speech playback."""
        # 1. If macOS 'say' binary is specified or injected by tests, use subprocess
        if self._say_binary:
            with self._lock:
                rate = self._rate
                voice = self._voice

            cmd: List[str] = [self._say_binary]
            if rate:
                cmd.extend(["-r", str(rate)])
            if voice:
                cmd.extend(["-v", voice])
            cmd.extend(["--", text])

            with self._lock:
                self._speaking = True
                try:
                    self._process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except FileNotFoundError:
                    logger.error("The 'say' command was not found on this system.")
                    self._speaking = False
                    self._process = None
                    return
                except Exception as e:
                    logger.error("Failed to start TTS process: %s", e)
                    self._speaking = False
                    self._process = None
                    return

            # Wait outside the lock so stop() can terminate the process concurrently
            try:
                self._process.wait()
            except Exception:
                pass
            finally:
                with self._lock:
                    self._speaking = False
                    self._process = None
            return

        # 2. Windows / worker branch
        if not self._worker_thread or not self._worker_thread.is_alive():
            self._start_worker()

        with self._lock:
            rate = self._rate
            voice = self._voice

        done = threading.Event()
        self._queue.put((text, rate, voice, done))
        # Wait up to 15 seconds for phrase playback
        done.wait(timeout=15.0)

    def stop(self) -> None:
        """Kills any ongoing speech immediately."""
        self._stop_event.set()

        # Terminate subprocess if active
        with self._lock:
            if self._process is not None:
                try:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=0.2)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=0.1)
                except Exception as e:
                    logger.debug("Error stopping speech process: %s", e)
                finally:
                    self._process = None
                    self._speaking = False

        # Clear queued requests
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item and len(item) > 3 and item[3] is not None:
                    item[3].set()
                self._queue.task_done()
            except queue.Empty:
                break

        with self._lock:
            self._speaking = False

    def is_speaking(self) -> bool:
        """Returns True if speech is currently playing."""
        with self._lock:
            if self._process is not None:
                if self._process.poll() is None:
                    return True
            return self._speaking

    def set_rate(self, rate: int) -> None:
        """Sets the speech rate in words per minute."""
        with self._lock:
            if rate <= 0:
                raise ValueError("Rate must be a positive integer")
            self._rate = int(rate)

    def get_rate(self) -> int:
        """Returns the current speech rate."""
        with self._lock:
            return self._rate

    def set_voice(self, voice: Optional[str]) -> None:
        """Sets the voice name or identifier."""
        with self._lock:
            self._voice = voice.strip() if voice and voice.strip() else None

    def get_voice(self) -> Optional[str]:
        """Returns current voice name."""
        with self._lock:
            return self._voice

    def close(self) -> None:
        """Stops any active speech process and shuts down worker."""
        self._shutdown = True
        self.stop()
        self._queue.put(None)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
