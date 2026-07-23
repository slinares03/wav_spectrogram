"""wav_spectrogram — a spectrogram player and waterfall for WAV files.

Handles both real audio and 2-channel I/Q WAVs, and shares the dB scaling and
contrast conventions used by spectrogram-labeling-tool and vita49_pipeline, so
images are comparable across all three.
"""

from .stft import StftEngine, analyze, psd_db, spectrogram_of
from .wav_source import WavSource

__all__ = ["WavSource", "StftEngine", "analyze", "psd_db", "spectrogram_of"]
