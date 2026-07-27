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

stdlib `wave` is the baseline decoder (PCM 8/16/24/32), with a small built-in
RIFF parser covering the float and WAVE_FORMAT_EXTENSIBLE files it rejects. If
`soundfile` is installed it is used instead, for wider format coverage still.
"""

from __future__ import annotations

import re
import wave
from pathlib import Path
from typing import Iterator

import numpy as np

# Filenames that look like SDR captures, e.g. "capture_433.92MHz_iq.wav"
_IQ_HINT = re.compile(r"(^|[_\-.])(iq|baseband|complex)([_\-.]|$)", re.IGNORECASE)


def _pcm_to_float(raw: bytes, sampwidth: int, n_channels: int,
                  is_float: bool = False) -> np.ndarray:
    """Decode interleaved PCM bytes to float32 in [-1, 1), shape (frames, ch)."""
    if is_float:
        # IEEE float samples are already in [-1, 1]; only the width varies.
        dtype = {4: "<f4", 8: "<f8"}.get(sampwidth)
        if dtype is None:
            raise ValueError(f"Unsupported float sample width: {sampwidth} bytes")
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    elif sampwidth == 1:
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


class _RiffReader:
    """Minimal WAV reader for the formats stdlib `wave` rejects.

    `wave` only decodes wFormatTag 1 (integer PCM), so float files (tag 3) and
    WAVE_FORMAT_EXTENSIBLE (tag 0xFFFE, emitted by many SDR and DAW tools) fail
    with "unknown format". This walks the RIFF chunks directly and exposes the
    same handful of methods WavSource uses, plus `is_float`.
    """

    _PCM, _FLOAT, _EXTENSIBLE = 0x0001, 0x0003, 0xFFFE

    def __init__(self, path: str):
        self._f = open(path, "rb")
        try:
            self._parse_header()
        except Exception:
            self._f.close()
            raise

    def _parse_header(self) -> None:
        head = self._f.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError("not a RIFF/WAVE file")

        fmt = None
        while True:
            hdr = self._f.read(8)
            if len(hdr) < 8:
                raise ValueError("WAV has no data chunk")
            cid, size = hdr[:4], int.from_bytes(hdr[4:8], "little")
            if cid == b"fmt ":
                fmt = self._f.read(size)
                self._f.seek(size % 2, 1)       # chunks are word-aligned
            elif cid == b"data":
                if fmt is None:
                    raise ValueError("WAV data chunk precedes fmt chunk")
                self._data_start = self._f.tell()
                self._data_size = size
                break
            else:
                self._f.seek(size + size % 2, 1)

        tag, self._n_channels, self._rate = int.from_bytes(fmt[0:2], "little"), \
            int.from_bytes(fmt[2:4], "little"), int.from_bytes(fmt[4:8], "little")
        bits = int.from_bytes(fmt[14:16], "little")
        if tag == self._EXTENSIBLE:
            # The real format lives in the first 2 bytes of the GUID subformat.
            if len(fmt) < 26:
                raise ValueError("truncated WAVE_FORMAT_EXTENSIBLE header")
            tag = int.from_bytes(fmt[24:26], "little")
        if tag not in (self._PCM, self._FLOAT):
            raise ValueError(f"unsupported WAV format tag: {tag}")

        self.is_float = tag == self._FLOAT
        self._sampwidth = bits // 8
        if self._sampwidth == 0 or self._n_channels == 0:
            raise ValueError("invalid WAV fmt chunk")
        self._frame_size = self._sampwidth * self._n_channels
        # A streamed file can declare size 0 or overshoot; trust the file length.
        avail = max(0, self._f.seek(0, 2) - self._data_start)
        if not 0 < self._data_size <= avail:
            self._data_size = avail
        self._f.seek(self._data_start)
        self._pos = 0

    def getframerate(self) -> int:
        return self._rate

    def getnchannels(self) -> int:
        return self._n_channels

    def getnframes(self) -> int:
        return self._data_size // self._frame_size

    def getsampwidth(self) -> int:
        return self._sampwidth

    def readframes(self, n: int) -> bytes:
        want = min(n * self._frame_size, self._data_size - self._pos)
        if want <= 0:
            return b""
        raw = self._f.read(want)
        self._pos += len(raw)
        return raw[:len(raw) - len(raw) % self._frame_size]

    def close(self) -> None:
        self._f.close()


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
            try:
                self._wav = wave.open(str(self.path), "rb")
                self._is_float = False
            except wave.Error:
                # Float / WAVEX files that stdlib `wave` refuses to open.
                self._wav = _RiffReader(str(self.path))
                self._is_float = self._wav.is_float
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
        return _pcm_to_float(raw, self._sampwidth, self.n_channels, self._is_float)

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
