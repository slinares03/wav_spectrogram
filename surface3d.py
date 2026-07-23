"""3-D ribbon view: the spectrogram window drawn as amplitude relief.

Axis convention (the default; reconfigurable at runtime, see `configure`):

    X = time      across the page
    Y = amplitude into the page   (inverted: low on the left)
    Z = frequency height

Any of the three data dimensions — time, amplitude, frequency — can be moved
onto any of the three spatial axes, and any spatial axis can be flipped, without
rebuilding the view. `DEFAULT_SRC` / `DEFAULT_FLIP` capture the layout above.

Why ribbons and not a surface
-----------------------------
`plot_surface` is unusable for playback: it rebuilds and re-sorts one quad per
cell, measuring ~250 ms/frame at a coarse 60x50 and ~5.9 s at export resolution.
A `Poly3DCollection` of one filled trace per time slice is a single artist whose
geometry we can replace wholesale, which measures ~24 ms/frame at 48x192 — the
classic "stacked spectra" waterfall look, and fast enough to play.

Why the time axis is relative
-----------------------------
Playback only reaches that 24 ms by blitting: a full figure draw re-renders the
3-D panes, grid and tick labels and costs ~250 ms. Blitting requires the cached
background to stay valid, so the axes limits must not move. The X axis is
therefore *time within the visible window* (0 at the oldest edge), which is
constant, rather than absolute file time, which would change every frame and
invalidate the background. Absolute position is reported by the player's status
readout and scrub bar instead.

Rotating the camera invalidates the background too — the owner must call
`invalidate()` and recapture after any view change.
"""

from __future__ import annotations

import numpy as np

# The three data dimensions that can be assigned to spatial axes, and the
# default assignment / flip state that reproduces the convention above.
SOURCES = ("time", "amp", "freq")
DEFAULT_SRC = {"x": "time", "y": "amp", "z": "freq"}
DEFAULT_FLIP = {"x": False, "y": True, "z": False}


class RibbonView:
    """Filled spectrum traces stacked along time, drawn into a 3-D axes.

    Args:
        ax:        an Axes3D to draw into.
        freqs:     frequency axis (Hz) of the incoming windows.
        window_s:  seconds of history a window covers (sets the X limit).
        vmin/vmax: dB limits — also the Z limits, so contrast and height agree.
        n_traces:  time slices drawn per frame. The dominant cost; see module
                   docstring for the measured trade.
        n_bins:    frequency points per trace.
    """

    def __init__(self, ax, freqs, window_s: float, vmin: float, vmax: float,
                 cmap: str = "magma", n_traces: int = 48, n_bins: int = 192,
                 is_iq: bool = False, title: str = ""):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.ax = ax
        self.n_traces = int(n_traces)
        self.vmin, self.vmax = float(vmin), float(vmax)
        self.window_s = float(window_s)
        self.is_iq = bool(is_iq)
        # Which data source each spatial axis draws, and whether it is flipped.
        # Copied so the module-level defaults stay pristine and per-instance
        # reconfiguration never leaks between views.
        self.src = dict(DEFAULT_SRC)
        self.flip = dict(DEFAULT_FLIP)

        # Frequency is decimated once, up front: the bin grid never changes, so
        # the reduced axis and the averaging groups can both be precomputed.
        n_bins = min(int(n_bins), freqs.size)
        self._f_group = max(freqs.size // n_bins, 1)
        self._n_f = freqs.size // self._f_group
        self._n_keep = self._n_f * self._f_group
        self.freqs = freqs[: self._n_keep].reshape(self._n_f, self._f_group).mean(axis=1)

        # X positions of the traces: oldest at 0, newest at window_s.
        self._times = np.linspace(0.0, self.window_s, self.n_traces)
        self._cmap = plt.get_cmap(cmap)

        # Each trace is a closed polygon: the spectrum, then back along the
        # floor, so it reads as a solid ribbon and occludes what is behind it.
        # Frequency is the *vertical* (Z) axis, so this closed frequency array
        # supplies the Z coordinate; amplitude runs into the page on Y.
        self._z_freq = np.concatenate([self.freqs[:1], self.freqs, self.freqs[-1:]])

        # Colour runs along *time*, dark (oldest) to bright (newest), and is
        # fixed for the life of the view. Amplitude is already the Z axis, so
        # re-encoding it as colour is redundant — worse, on steady material
        # every trace shares the same peak and the whole plot flattens into one
        # colour. A recency ramp instead separates overlapping traces, which is
        # the thing that actually makes a stacked waterfall readable.
        # Fills are opaque: they are what occludes the traces behind them.
        self._last_window = None            # last data window, for live remaps
        shade = self._cmap(np.linspace(0.15, 0.95, self.n_traces))
        self.collection = Poly3DCollection(
            self._verts(np.full((self.n_traces, self._n_f), self.vmin)),
            facecolors=shade, linewidths=0.5, edgecolors=(0, 0, 0, 0.55))
        ax.add_collection3d(self.collection)

        self._apply_layout()
        if title:
            ax.set_title(title, fontsize=10)
        ax.view_init(elev=32.0, azim=-58.0)

    # -- axis mapping ------------------------------------------------------

    def _source_range_label(self, src):
        """(lo, hi, axis label, EngFormatter unit or None) for a data source."""
        if src == "time":
            return 0.0, self.window_s, "Time in window", "s"
        if src == "freq":
            return (self.freqs[0], self.freqs[-1],
                    "RF frequency" if self.is_iq else "Frequency", "Hz")
        return self.vmin, self.vmax, "Amplitude [dB]", None      # "amp"

    def _apply_layout(self):
        """Push the current src/flip mapping onto the axes: limits, labels and
        tick formatters, one spatial axis at a time."""
        from matplotlib.ticker import EngFormatter, ScalarFormatter

        for axis in ("x", "y", "z"):
            lo, hi, label, unit = self._source_range_label(self.src[axis])
            if self.flip[axis]:
                lo, hi = hi, lo
            getattr(self.ax, f"set_{axis}lim")(lo, hi)
            getattr(self.ax, f"set_{axis}label")(label)
            fmt = EngFormatter(unit) if unit else ScalarFormatter()
            getattr(self.ax, f"{axis}axis").set_major_formatter(fmt)

    # -- geometry ----------------------------------------------------------

    def _coord(self, src, t, amp):
        """The coordinate array a spatial axis contributes for one trace.

        A trace lives at a fixed time `t`; within it frequency and amplitude
        vary bin by bin. `amp` is the closed amplitude polygon (already padded
        back to the vmin floor at both ends) and `self._z_freq` the matching
        closed frequency array, so all three sources share one length.
        """
        if src == "time":
            return np.full(amp.size, t)
        if src == "freq":
            return self._z_freq
        return amp                                  # "amp"

    def _verts(self, z: np.ndarray) -> list:
        """(n_traces, n_f) dB -> one closed polygon per trace, placed on the
        spatial axes according to the current src mapping."""
        floor = self.vmin
        sx, sy, sz = self.src["x"], self.src["y"], self.src["z"]
        out = []
        for i, t in enumerate(self._times):
            amp = np.concatenate([[floor], z[i], [floor]])
            out.append(np.column_stack([self._coord(sx, t, amp),
                                        self._coord(sy, t, amp),
                                        self._coord(sz, t, amp)]))
        return out

    def _reduce(self, psd_win: np.ndarray) -> np.ndarray:
        """Window (time, freq) -> (n_traces, n_f), averaged in both axes.

        Averaging rather than sampling matters here: at 48 traces from a window
        of several hundred rows, plain slicing would drop most of the audio and
        make short transients flicker in and out between frames.
        """
        p = psd_win[:, : self._n_keep]
        n_t = p.shape[0]
        if n_t >= self.n_traces:
            g = n_t // self.n_traces
            p = p[: g * self.n_traces].reshape(self.n_traces, g, -1).mean(axis=1)
        else:                                   # window shorter than the trace
            idx = np.clip((np.arange(self.n_traces) * n_t) // self.n_traces, 0, n_t - 1)
            p = p[idx]                          # count: repeat rows instead
        return p.reshape(self.n_traces, self._n_f, self._f_group).mean(axis=2)

    # -- update ------------------------------------------------------------

    def set_window(self, psd_win: np.ndarray) -> None:
        """Replace the geometry from a (time, freq) dB window, oldest first."""
        self._last_window = psd_win        # kept so a remap can rebuild in place
        z = np.clip(self._reduce(psd_win), self.vmin, self.vmax)
        self.collection.set_verts(self._verts(z))
        # Face colours are static (a recency ramp set in __init__), so geometry
        # is the only thing that changes per frame.

    def set_clim(self, vmin: float, vmax: float) -> None:
        self.vmin, self.vmax = float(vmin), float(vmax)
        self._apply_layout()          # rescale whichever axis carries amplitude

    # -- runtime reconfiguration -------------------------------------------

    def swap_axes(self, a: str, b: str) -> None:
        """Exchange the data sources on two spatial axes ('x'/'y'/'z').

        Flips stay with the physical axis, so swapping only moves which quantity
        each axis shows, not its orientation.
        """
        self.src[a], self.src[b] = self.src[b], self.src[a]
        self._reconfigure()

    def flip_axis(self, axis: str) -> None:
        """Invert one spatial axis ('x'/'y'/'z')."""
        self.flip[axis] = not self.flip[axis]
        self._reconfigure()

    def reset_axes(self) -> None:
        """Return to the default mapping and orientation."""
        self.src = dict(DEFAULT_SRC)
        self.flip = dict(DEFAULT_FLIP)
        self._reconfigure()

    def describe(self) -> str:
        """One-line summary of the current mapping, e.g. 'X:time  Y:-amp  Z:freq'."""
        return "  ".join(
            f"{a.upper()}:{'-' if self.flip[a] else ''}{self.src[a]}"
            for a in ("x", "y", "z"))

    def _reconfigure(self) -> None:
        """Re-apply the mapping to both the axes and the existing geometry.

        The geometry has to be rebuilt because which coordinate each vertex
        column feeds changed; the caller owns the blit background and must
        invalidate it, since limits and labels have moved.
        """
        self._apply_layout()
        z = np.clip(self._reduce(self._last_window), self.vmin, self.vmax) \
            if self._last_window is not None \
            else np.full((self.n_traces, self._n_f), self.vmin)
        self.collection.set_verts(self._verts(z))

    def project(self) -> None:
        """Re-run the 3-D projection.

        The axes' own draw normally does this. When blitting we call
        `draw_artist` directly, which does not — skipping it silently renders
        the *previous* frame's projection, so the display freezes while the
        data underneath keeps changing.
        """
        self.collection.do_3d_projection()
