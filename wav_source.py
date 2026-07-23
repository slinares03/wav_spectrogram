"""Streaming WAV reader that yields blocks of audio or IQ samples.

Reads in blocks rather than loading the file, because the point of this package
is a *live* display: a 40-minute WAV should start drawing immediately and cost
the same memory as a 4-second one.

Two interpretations of a WAV are supported, because both show up in this project:

  audio  — real-valued samples (mono, or stereo mixed down). Spectrum is
           one-sided, 0 .. fs/2.
  iq     — a 2-channel file where L = I and R = Q, the usual way SDR tools dump
           complex baseband into a WAV. Spectrum is two-sided and centred on
           whatever RF frequency the capture was tuned to (--center).

`mode="auto"` picks iq for 2-channel files only when told to; a plain stereo
music file would be nonsense as IQ, so auto resolves 2-channel to audio unless
--center was given or the filename hints at IQ. Be explicit with --mode when it
matters.

stdlib `wave` is the baseline decoder (PCM 8/16/24/32). If `soundfile` is
installed it is used instead, which additionally handles float WAV and WAVEX.
"""

from __future__ import annotations

import re
import wave
from pathlib import Path
from typing import Iterator

import numpy as np

# Filenames that look like SDR captures, e.g. "capture_433.92MHz_iq.wav"
_IQ_HINT = re.compile(r"(^|[_\-.])(iq|baseband|complex)([_\-.]|$)", re.IGNORECASE)


def _pcm_to_float(raw: bytes, sampwidth: int, n_channels: int) -> np.ndarray:
    """Decode interleaved PCM bytes to float32 in [-1, 1), shape (frames, ch)."""
    if sampwidth == 1:
        # 8-bit WAV is unsigned, offset by 128 — every other width is signed.
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 3:
        # 24-bit has no numpy dtype: widen each 3-byte group into a 4-byte
        # little-endian int32 (sign in the top byte), then scale.
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        wide = np.zeros((len(b), 4), dtype=np.uint8)
        wide[:, 1:] = b                       # shift left 8 bits -> sign lands in MSB
        data = wide.view("<i4").ravel().astype(np.float32) / 2147483648.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported PCM sample width: {sampwidth} bytes")
    return data.reshape(-1, n_channels)


class WavSource:
    """A WAV file presented as a stream of sample blocks.

    Attributes:
        sample_rate  Hz
        n_channels   channels in the file
        n_frames     total sample frames
        duration     seconds
        is_iq        True if blocks() yields complex64, False for float32
        center_hz    frequency the spectrum is centred on (0 for audio)
    """

    def __init__(self, path, mode: str = "auto", center_hz: float = 0.0):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

        self._sf = None
        try:                                   # optional, better format coverage
            import soundfile
            self._sf = soundfile.SoundFile(str(self.path))
            self.sample_rate = int(self._sf.samplerate)
            self.n_channels = int(self._sf.channels)
            self.n_frames = int(self._sf.frames)
            self._sampwidth = None
        except ImportError:
            self._wav = wave.open(str(self.path), "rb")
            self.sample_rate = self._wav.getframerate()
            self.n_channels = self._wav.getnchannels()
            self.n_frames = self._wav.getnframes()
            self._sampwidth = self._wav.getsampwidth()

        self.duration = self.n_frames / self.sample_rate
        self.center_hz = float(center_hz)
        self.is_iq = self._resolve_mode(mode)
        if self.is_iq and self.n_channels < 2:
            raise ValueError("IQ mode needs a 2-channel WAV (L=I, R=Q); "
                             f"{self.path.name} has {self.n_channels}")

    def _resolve_mode(self, mode: str) -> bool:
        if mode == "iq":
            return True
        if mode == "audio":
            return False
        if mode != "auto":
            raise ValueError(f"mode must be auto|audio|iq, got {mode!r}")
        # Auto: only claim IQ on a 2-channel file with corroborating evidence,
        # so ordinary stereo audio is never misread as complex baseband.
        return self.n_channels == 2 and (
            self.center_hz != 0.0 or bool(_IQ_HINT.search(self.path.stem)))

    def _read_frames(self, n: int) -> np.ndarray | None:
        """Next `n` sample frames as (frames, channels) float32, or None at EOF."""
        if self._sf is not None:
            data = self._sf.read(n, dtype="float32", always_2d=True)
            return None if len(data) == 0 else data
        raw = self._wav.readframes(n)
        if not raw:
            return None
        return _pcm_to_float(raw, self._sampwidth, self.n_channels)

    def blocks(self, block_size: int) -> Iterator[np.ndarray]:
        """Yield 1-D blocks: complex64 when is_iq, else float32 mono.

        Multi-channel audio is averaged to mono — a spectrogram of a channel sum
        is what you want for "what frequencies are in this file", and it halves
        the FFT work versus drawing both.
        """
        while True:
            frames = self._read_frames(block_size)
            if frames is None:
                return
            if self.is_iq:
                yield (frames[:, 0] + 1j * frames[:, 1]).astype(np.complex64)
            elif self.n_channels == 1:
                yield frames[:, 0].astype(np.float32)
            else:
                yield frames.mean(axis=1).astype(np.float32)

    def close(self) -> None:
        if self._sf is not None:
            self._sf.close()
        else:
            self._wav.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def describe(self) -> str:
        kind = "IQ (complex baseband)" if self.is_iq else "audio (real)"
        return (f"{self.path.name}: {kind}, {self.sample_rate/1e3:.1f} kHz, "
                f"{self.n_channels} ch, {self.duration:.2f} s")
