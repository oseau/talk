#!/usr/bin/env python3
"""
ASR Example using common libraries.

This example demonstrates how to use FunASR's SenseVoice model.

Dependencies:
- brew install portaudio

Usage:
    uv run python main.py

Features:
- Loads SenseVoice ASR model
- Live microphone recording with real-time transcription
"""

import asyncio
import logging
import os
import sys
import time
from typing import Any, Optional

import demoji
import numpy as np
import sounddevice as sd
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from modelscope.utils.file_utils import get_modelscope_cache_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # Override logging configs set by import libs
)
logger = logging.getLogger(os.path.basename(__file__).replace(".py", ""))


class AudioRecorder:
    """Simple audio recorder using sounddevice with async context manager."""

    def __init__(
        self,
        samplerate: int = 16000,
        channels: int = 1,
        blocksize: int = 1024,
        dtype: type = np.int16,
        device: Optional[str] = None,
    ) -> None:
        """Initialize audio recorder.

        Args:
            samplerate: Audio sample rate (default: 16000 Hz for ASR)
            channels: Number of audio channels (default: 1 for mono)
            blocksize: Number of samples per block (default: 1024)
            dtype: Audio data type (default: int16)
            device: Input device name or index (default: None for system default)
        """
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.dtype = dtype
        self.device = device
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.stream: Optional[sd.RawInputStream] = None
        self.is_running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def _audio_callback(self, indata, frames, callback_time, status):
        """Audio callback function for sounddevice."""
        if status:
            logger.warning("Audio callback status: %s", status)

        # Queue the audio data safely
        if self.loop:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, bytes(indata))

    async def __aenter__(self) -> "AudioRecorder":
        """Async context manager entry point.

        Returns:
            AudioRecorder: Self for async iterator usage
        """
        try:
            # Create audio input stream
            self.stream = sd.RawInputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype=self.dtype,
                callback=self._audio_callback,
            )

            # Get current event loop
            self.loop = asyncio.get_running_loop()

            self.stream.start()
            self.is_running = True
            logger.info(
                "Recording started: device=%s, samplerate=%d, channels=%d",
                self.device,
                self.samplerate,
                self.channels,
            )

            return self

        except Exception as e:
            logger.error("Failed to start recording: %s", e)
            await self.stop()
            raise

    def __aiter__(self) -> "AudioRecorder":
        """Async iterator entry point.

        Returns:
            AudioRecorder: Self for async iteration
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit point with guaranteed cleanup.

        Args:
            exc_type: Exception type (None if no exception)
            exc_val: Exception value (None if no exception)
            exc_tb: Exception traceback (None if no exception)
        """
        await self.stop()
        logger.info("Recording stopped")

    async def __anext__(self) -> bytes:
        """Async iterator implementation."""
        while self.is_running:
            try:
                # Get audio data from queue
                audio_data = await self.queue.get()
                return audio_data
            except asyncio.CancelledError:
                # Handle Ctrl+C here gracefully - stop iteration
                break
            except Exception as e:
                logger.error("Recording error: %s", e)
                break

        # Stop iteration
        raise StopAsyncIteration

    async def stop(self) -> None:
        """Stop recording and cleanup."""
        self.is_running = False
        if self.stream:
            self.stream.close()
            self.stream = None

        # Clear the queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class SenseVoiceASR:
    """SenseVoice ASR implementation.

    Minimal dependencies and easy to customize.
    """

    def __init__(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        device: str = "cpu",
        language: str = "auto",
        use_itn: bool = True,
        batch_size: int = 1,
    ) -> None:
        """Initialize SenseVoice ASR.

        Args:
            model_name: Model name or path (default: "iic/SenseVoiceSmall")
            device: Device to use ('cpu', 'cuda', 'mps', etc.)
            language: Language code ('auto', 'zh', 'en', 'ja', 'ko', etc.)
            use_itn: Use inverse text normalization (default: True)
            batch_size: Batch size for processing (default: 1)
        """
        self.model_name = model_name
        self.device = device
        self.language = language
        self.use_itn = use_itn
        self.batch_size = batch_size
        self.model: Optional[Any] = None
        self.ready = False

        # Auto-initialize by default
        self.initialize()

    def initialize(self) -> bool:
        """Initialize the SenseVoice model.

        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Loading model: %s on device: %s", self.model_name, self.device)
            start_time = time.time()

            path = os.path.join(
                get_modelscope_cache_dir(),
                "models",
                *self.model_name.split("/"),
            )
            # works truly offline after first run
            kwargs = {"model_path": path} if os.path.exists(path) else {}

            # Load FunASR model
            self.model = AutoModel(
                model=self.model_name,
                device=self.device,
                batch_size=self.batch_size,
                disable_update=True,  # Disable auto-updates for stability
                **kwargs,
            )

            load_time = time.time() - start_time
            logger.info(
                "Model loaded successfully in %.2f seconds",
                load_time,
            )
            self.ready = True
            return True

        except Exception as e:
            logger.error("Failed to initialize model: %s", e)
            self.ready = False
            return False

    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes (int16 format)

        Returns:
            str: Transcribed text
        """
        if not self.is_ready():
            raise RuntimeError("ASR model not initialized")

        try:
            # Run transcription with timeout to make it interruptible
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: self.model.generate(
                        input=audio_data,
                        language=self.language,
                        use_itn=self.use_itn,
                    ),
                ),
                timeout=10.0,  # 10 second timeout
            )
            # Post-process the result
            if result:
                result = rich_transcription_postprocess(result[0]["text"]).strip()
                return demoji.replace(result)
            return ""

        except asyncio.TimeoutError:
            logger.warning("Transcription timeout")
            return ""
        except asyncio.CancelledError:
            # Handle Ctrl+C gracefully during transcription
            return ""
        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return ""

    def is_ready(self) -> bool:
        """Check if the model is ready for transcription."""
        return self.ready and self.model is not None

    def release(self) -> bool:
        """Release model resources."""
        try:
            logger.info("Releasing ASR resources")
            self.model = None
            self.ready = False
            return True
        except Exception as e:
            logger.error("Failed to release resources: %s", e)
            return False


def list_audio_devices() -> list:
    """List available audio devices."""
    try:
        devices = sd.query_devices()
        logger.info("Available audio devices:")

        input_devices = []
        default_input = sd.default.device[0]

        for i, device in enumerate(devices):
            device_type = ""
            if device["max_input_channels"] > 0:
                device_type += "input"
            if device["max_output_channels"] > 0:
                device_type += "output"

            if device_type:
                is_default_input = i == default_input and "input" in device_type
                is_default = " (default)" if is_default_input else ""
                device_info = f"[{i}] {device['name']} - {device_type}{is_default}"
                logger.info("  %s", device_info)

                if device["max_input_channels"] > 0:
                    input_devices.append(i)

        if input_devices:
            logger.info("Use device index %d to select input device", input_devices[0])
        else:
            logger.warning("No input devices found!")

        return devices
    except Exception as e:
        logger.error("Failed to list audio devices: %s", e)
        return []


async def test_live_recording(asr: SenseVoiceASR) -> None:
    """Test ASR with live microphone recording using context manager."""

    # Initialize audio recorder
    recorder = AudioRecorder(
        samplerate=16000,  # ASR standard sample rate
        channels=1,  # Mono
        blocksize=1024,  # Small blocks for low latency
        dtype=np.int16,  # Standard for ASR
    )

    # Buffer to accumulate audio for transcription
    audio_buffer = np.array([], dtype=np.int16)
    buffer_duration = 3.0  # Process every 3 seconds of audio
    buffer_samples = int(buffer_duration * 16000)  # samples

    try:
        async with recorder as audio_stream:
            async for audio_chunk in audio_stream:
                # Convert bytes to numpy array
                chunk_array = np.frombuffer(audio_chunk, dtype=np.int16)

                # Accumulate audio
                audio_buffer = np.concatenate([audio_buffer, chunk_array])

                # When we have enough audio, process it
                if len(audio_buffer) >= buffer_samples:
                    # Extract the required samples
                    speech_segment = audio_buffer[:buffer_samples]
                    audio_bytes = speech_segment.tobytes()

                    try:
                        # Transcribe the audio
                        start_time = time.time()
                        text = await asr.transcribe(audio_bytes)
                        transcription_time = time.time() - start_time

                        if text:
                            logger.info(
                                "Transcription (%.2fs): %s",
                                transcription_time,
                                text,
                            )
                        else:
                            logger.debug("No speech detected in audio segment")

                    except Exception as e:
                        logger.error("Transcription error: %s", e)

                    # Keep half of the buffer for overlap
                    audio_buffer = audio_buffer[buffer_samples // 2 :]

    except Exception as e:
        logger.error("Recording error: %s", e)
        # Note: recorder.__aexit__ already called for cleanup


async def main() -> None:
    """Main function demonstrating ASR usage."""

    # List available audio devices
    list_audio_devices()

    # Initialize ASR
    asr = SenseVoiceASR(
        model_name="iic/SenseVoiceSmall",  # Small model for faster loading
        device="cpu",  # Use CPU, change to "cuda" or "mps" if available
        language="auto",  # Auto-detect language
        use_itn=True,  # Use inverse text normalization
        batch_size=1,
    )

    if not asr.is_ready():
        return

    try:
        await test_live_recording(asr)
    except Exception as e:
        logger.error("Exception in main: %s", e)

    finally:
        # Cleanup
        asr.release()


if __name__ == "__main__":
    # Simple interrupt handling
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        sys.exit(1)
