"""Streaming STFT: sample blocks in, spectrogram rows out.

The dB convention here is deliberately identical to
spectrogram-labeling-tool/spectrogram/utils.py and vita49_pipeline —
`np.fft.fft(..., norm="forward")` followed by `10*log10(10*|X|**2)` — so a
fixed --vmin/--vmax chosen for one of those tools means the same thing here, and
images from all three are visually comparable.

Rows are power-averaged in groups of `avg` (the labeling tool's n_avg_frames).
Averaging power rather than complex spectra is the right choice for a live view:
it smooths the noise floor without cancelling a signal whose phase is drifting
between frames.

The engine carries a tail buffer across pushes, so hop/overlap is continuous and
frames straddling a block boundary are not lost.
"""

from __future__ import annotations

import numpy as np


def psd_db(power: np.ndarray) -> np.ndarray:
    """Power -> dB, matching spectrogram.utils.psd (10*log10(10*|x|^2))."""
    return 10.0 * np.log10(10.0 * power + 1e-12)


class StftEngine:
    """Overlapping STFT over an unbounded stream of samples.

    Args:
        sample_rate: Hz.
        n_fft:       FFT length (frequency bins; n_fft//2+1 for real audio).
        hop:         samples advanced between frames. hop < n_fft gives overlap.
        is_iq:       complex input -> two-sided fftshifted spectrum.
        center_hz:   added to the frequency axis (IQ tuning frequency).
        avg:         power-average this many FFT frames into each output row.
    """

    def __init__(self, sample_rate: float, n_fft: int = 1024, hop: int | None = None,
                 is_iq: bool = False, center_hz: float = 0.0, avg: int = 1):
        if n_fft < 16:
            raise ValueError("n_fft must be >= 16")
        self.sample_rate = float(sample_rate)
        self.n_fft = int(n_fft)
        self.hop = int(hop if hop else n_fft // 4)   # 75% overlap default
        self.is_iq = bool(is_iq)
        self.avg = max(int(avg), 1)
        self.window = np.hanning(self.n_fft).astype(np.float32)

        if is_iq:
            self.freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1 / sample_rate)) + center_hz
        else:
            self.freqs = np.fft.rfftfreq(n_fft, d=1 / sample_rate)
        self.n_bins = self.freqs.size

        dtype = np.complex64 if is_iq else np.float32
        self._tail = np.zeros(0, dtype=dtype)
        self._pending = np.zeros((0, self.n_bins), dtype=np.float32)  # un-averaged power
        self._frames_emitted = 0

    @property
    def seconds_per_row(self) -> float:
        """Wall-clock audio time each output row represents."""
        return self.hop * self.avg / self.sample_rate

    def push(self, samples: np.ndarray) -> np.ndarray:
        """Consume a block; return however many dB rows it completed.

        Returns shape (n_rows, n_bins); n_rows may be 0 when a block is shorter
        than one hop group. Row order is oldest-first.
        """
        buf = np.concatenate([self._tail, samples]) if self._tail.size else samples
        n_frames = 1 + (len(buf) - self.n_fft) // self.hop if len(buf) >= self.n_fft else 0

        if n_frames > 0:
            # Strided view of all frames at once — one batched FFT beats a loop.
            idx = np.arange(self.n_fft)[None, :] + self.hop * np.arange(n_frames)[:, None]
            frames = buf[idx] * self.window[None, :]
            if self.is_iq:
                spec = np.fft.fftshift(np.fft.fft(frames, axis=1, norm="forward"), axes=1)
            else:
                spec = np.fft.rfft(frames, axis=1, norm="forward")
            power = (np.abs(spec) ** 2).astype(np.float32)
            self._pending = np.concatenate([self._pending, power])
            self._tail = buf[n_frames * self.hop:]
        else:
            self._tail = buf

        # Emit only whole averaging groups; the remainder waits for more data.
        n_rows = len(self._pending) // self.avg
        if n_rows == 0:
            return np.zeros((0, self.n_bins), dtype=np.float32)

        used = n_rows * self.avg
        rows = self._pending[:used].reshape(n_rows, self.avg, self.n_bins).mean(axis=1)
        self._pending = self._pending[used:]
        self._frames_emitted += n_rows
        return psd_db(rows)

    def elapsed_seconds(self) -> float:
        """Audio time covered by rows emitted so far."""
        return self._frames_emitted * self.seconds_per_row


def analyze(source, n_fft: int = 1024, hop: int | None = None, avg: int = 1,
            max_rows: int = 40000, block_size: int = 1 << 18, progress=None):
    """Precompute a whole file for the player: (psd [time, freq], freqs, row_s).

    Seeking and reverse need the matrix in memory, so `max_rows` bounds it: if
    the requested settings would produce more rows than that, `avg` is raised
    until they fit. Raising avg (rather than dropping rows afterwards) keeps
    every sample contributing to the image instead of throwing audio away — a
    long file gets a smoother, slightly coarser view rather than a gappy one.
    """
    hop = hop or n_fft // 4
    est_rows = max(source.n_frames // (hop * max(avg, 1)), 1)
    if est_rows > max_rows:
        avg = int(np.ceil(source.n_frames / (hop * max_rows)))

    eng = StftEngine(source.sample_rate, n_fft=n_fft, hop=hop, is_iq=source.is_iq,
                     center_hz=source.center_hz, avg=avg)
    chunks, done = [], 0
    for blk in source.blocks(block_size):
        rows = eng.push(blk)
        if len(rows):
            chunks.append(rows)
        done += len(blk)
        if progress is not None:
            progress(done / max(source.n_frames, 1))
    if not chunks:
        raise RuntimeError("File too short to produce a single spectrogram row "
                           f"(needs >= {n_fft + hop * (avg - 1)} samples)")
    if progress is not None:
        progress(1.0)
    return np.concatenate(chunks), eng.freqs, eng.seconds_per_row


def spectrogram_of(source, n_fft: int = 1024, hop: int | None = None,
                   avg: int = 1, max_rows: int | None = None,
                   block_size: int = 1 << 16):
    """Whole-file spectrogram as (psd_dB [time, freq], freqs, times).

    Used by the PNG export path. Streams the file through the same engine the
    live view uses, so an exported image is exactly a still of the live display.
    If `max_rows` is set, rows are decimated by block-averaging at the end to
    bound memory and keep long files cheap.
    """
    eng = StftEngine(source.sample_rate, n_fft=n_fft, hop=hop,
                     is_iq=source.is_iq, center_hz=source.center_hz, avg=avg)
    chunks = [r for blk in source.blocks(block_size) if len((r := eng.push(blk)))]
    if not chunks:
        raise RuntimeError("File too short to produce a single spectrogram row "
                           f"(needs >= {n_fft + hop * (avg - 1)} samples)")
    psd = np.concatenate(chunks)

    if max_rows and len(psd) > max_rows:
        group = len(psd) // max_rows
        psd = psd[: group * max_rows].reshape(max_rows, group, -1).mean(axis=1)
        times = np.arange(max_rows) * eng.seconds_per_row * group
    else:
        times = np.arange(len(psd)) * eng.seconds_per_row
    return psd, eng.freqs, times
