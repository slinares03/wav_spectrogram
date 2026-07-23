"""Still-image export: whole-file spectrogram PNGs.

Two renderers, mirroring vita49_pipeline/capture_spectrogram.py so the outputs
drop straight into the existing dataset/labeling workflow:

  render_preview — annotated axes + colorbar, for humans.
  render_yolo    — bare square image, no decoration, for YOLO training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


MIN_SPAN_DB = 12.0      # never squeeze the colour range below this


def clim(psd: np.ndarray, vmin=None, vmax=None, mode: str = "floor"):
    """Colour limits in dB. Either limit may be pinned individually.

    mode="floor" (default) anchors the bottom just under the noise floor
    (median) and the top near the loudest content, so noise renders dark and
    signals stand out. mode="percentile" reproduces the labeling tool's 5/95
    rule exactly — right for its signal-dense captures, but on a typical WAV,
    where most bins are noise, 5/95 spans only the noise itself and the floor
    fills the whole colormap as speckle.
    """
    if mode == "percentile":
        lo, hi = np.percentile(psd, [5, 95])
    else:
        lo = float(np.median(psd)) - 2.0
        hi = float(np.percentile(psd, 99.9))
    if hi - lo < MIN_SPAN_DB:                 # near-silent or near-flat input
        hi = lo + MIN_SPAN_DB
    if vmin is not None:
        lo = float(vmin)
    if vmax is not None:
        hi = float(vmax)
    return float(lo), float(hi)


def _decimate(psd, freqs, times, max_t: int, max_f: int):
    """Block-average the matrix down to at most (max_t, max_f).

    A surface plot draws one quad per cell, so the full-resolution matrix
    (thousands x hundreds) is both unreadable and slow to render. Averaging
    rather than slicing keeps every sample contributing, so a narrow tone
    doesn't vanish between kept bins.
    """
    t_group = max(len(times) // max_t, 1)
    f_group = max(len(freqs) // max_f, 1)
    if t_group > 1:
        n = len(times) // t_group
        psd = psd[: n * t_group].reshape(n, t_group, -1).mean(axis=1)
        times = times[: n * t_group].reshape(n, t_group).mean(axis=1)
    if f_group > 1:
        n = psd.shape[1] // f_group
        psd = psd[:, : n * f_group].reshape(psd.shape[0], n, f_group).mean(axis=2)
        freqs = freqs[: n * f_group].reshape(n, f_group).mean(axis=1)
    return psd, freqs, times


def render_surface(psd, freqs, times, out_path: Path, title: str = "",
                   cmap: str = "magma", vmin=None, vmax=None, is_iq: bool = False,
                   clim_mode: str = "floor", max_t: int = 400, max_f: int = 200,
                   elev: float = 35.0, azim: float = -60.0, fig=None):
    """3-D surface: time across, frequency into the page, amplitude as height.

    The flat renderers encode amplitude as colour only; here amplitude is the
    vertical axis, which makes peak structure (harmonics, transients standing
    out of the floor) legible at a glance. Colour still tracks height, using
    the same dB limits as the 2-D images so the two read consistently.

    Returns the Figure. If `fig` is given the surface is drawn into it (for an
    embedded/interactive window); otherwise a headless figure is created and
    saved to `out_path`.
    """
    from matplotlib.ticker import EngFormatter

    lo, hi = clim(psd, vmin, vmax, clim_mode)
    psd, freqs, times = _decimate(psd, freqs, times, max_t, max_f)

    # Clip to the colour limits before plotting: without this a single loud
    # transient spikes off the top and flattens everything else into the floor.
    z = np.clip(psd, lo, hi)
    # meshgrid over (time, freq) -> X=time, Y=freq, Z=amplitude, matching the
    # (time, freq) index order of psd.
    x, y = np.meshgrid(times, freqs, indexing="ij")

    own_fig = fig is None
    if own_fig:
        # Explicit Agg canvas, never matplotlib.use(): see render_preview — the
        # global switch would close the live player window when it calls this.
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=(12, 7))
        FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap=cmap, vmin=lo, vmax=hi,
                    rstride=1, cstride=1,      # _decimate already sized the grid
                    linewidth=0, antialiased=False)
    ax.set_zlim(lo, hi)
    # Low frequencies at the *far* edge: on typical audio they carry the tallest
    # peaks, and at the near edge that wall hides everything behind it.
    ax.set_ylim(freqs[-1], freqs[0])
    ax.view_init(elev=elev, azim=azim)

    ax.xaxis.set_major_formatter(EngFormatter("s"))
    ax.yaxis.set_major_formatter(EngFormatter("Hz"))
    ax.set_xlabel("Time")
    ax.set_ylabel("RF frequency" if is_iq else "Frequency")
    ax.set_zlabel("Amplitude [dB]")
    ax.set_title(title)
    fig.tight_layout()

    if own_fig:
        fig.savefig(out_path, dpi=120)
    return fig


def render_preview(psd, freqs, times, out_path: Path, title: str = "",
                   cmap: str = "magma", vmin=None, vmax=None, is_iq: bool = False,
                   clim_mode: str = "floor"):
    # Build the figure through an explicit Agg canvas rather than pyplot +
    # matplotlib.use("Agg"): switching the global backend would tear down a live
    # GUI (the player's window) mid-run, which is why exporting used to close it.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.ticker import EngFormatter

    lo, hi = clim(psd, vmin, vmax, clim_mode)
    fig = Figure(figsize=(12, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    img = ax.imshow(psd, aspect="auto", origin="lower",
                    extent=[freqs[0], freqs[-1], times[0], times[-1]],
                    vmin=lo, vmax=hi, cmap=cmap, interpolation="nearest")
    ax.xaxis.set_major_formatter(EngFormatter("Hz"))
    ax.yaxis.set_major_formatter(EngFormatter("s"))
    ax.set_xlabel("RF frequency" if is_iq else "Frequency")
    ax.set_ylabel("Time")
    ax.set_title(title)
    fig.colorbar(img, ax=ax).set_label("[dB]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


def render_yolo(psd, out_path: Path, imgsz: int = 640, gray: bool = False,
                cmap: str = "magma", vmin=None, vmax=None, clim_mode: str = "floor"):
    from PIL import Image
    from matplotlib import colormaps

    lo, hi = clim(psd, vmin, vmax, clim_mode)
    norm = np.clip((psd - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    norm = np.flipud(norm)                    # row 0 = earliest time
    if gray:
        rgb = np.repeat((norm * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    else:
        # colormaps[...] rather than the get_cmap function removed in mpl 3.11
        rgb = (colormaps[cmap](norm)[:, :, :3] * 255).astype(np.uint8)
    # No mode arg: it is inferred from the array shape and is deprecated in
    # Pillow (removed in 13). The (H, W, 3) uint8 array is RGB either way.
    Image.fromarray(rgb).resize((imgsz, imgsz), Image.BILINEAR).save(out_path)
