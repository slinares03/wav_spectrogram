"""Live scrolling waterfall display.

Design notes:

* The display is a fixed-size ring buffer of rows, drawn with a single
  `imshow` whose data is replaced via `set_data`. Re-calling imshow (or using
  pcolormesh) per frame is what makes naive waterfalls stutter; one AxesImage
  updated in place holds 30 fps on a laptop.

* Newest row is at the top and older rows scroll down — the classic waterfall
  direction — so the y axis reads "age" in seconds, increasing downward.

* Contrast auto-tracks the visible window with export.clim (see there for the
  rule), smoothed by an EMA. Recomputing limits raw every frame makes the image
  visibly pulse whenever a loud event enters or leaves the window; the EMA keeps
  it stable while still adapting to a drifting noise floor. Pass fixed
  vmin+vmax to disable adaptation entirely — do that when comparing images
  across a dataset.
"""

from __future__ import annotations

import time

import numpy as np

from .export import clim as _clim


class Waterfall:
    """Rolling spectrogram image driven by rows from an StftEngine.

    Args:
        freqs:        frequency axis (Hz) for the x extent.
        seconds:      how much history the window shows.
        row_seconds:  audio seconds per row (StftEngine.seconds_per_row).
        cmap:         matplotlib colormap name.
        vmin, vmax:   fixed dB limits; None (default) = adaptive.
        is_iq:        labels the axis as RF vs audio frequency.
        title:        window/axes title.
        clim_mode:    "floor" or "percentile" (see export.clim).
    """

    _ADAPT_ALPHA = 0.15        # EMA weight for new contrast estimates

    def __init__(self, freqs: np.ndarray, seconds: float, row_seconds: float,
                 cmap: str = "magma", vmin: float | None = None,
                 vmax: float | None = None, is_iq: bool = False, title: str = "",
                 clim_mode: str = "floor"):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import EngFormatter

        self.freqs = freqs
        self.row_seconds = row_seconds
        self.n_rows = max(int(round(seconds / row_seconds)), 2)
        self.clim_mode = clim_mode
        self._vmin, self._vmax = vmin, vmax
        self.fixed_clim = vmin is not None and vmax is not None
        self._clim = (float(vmin), float(vmax)) if self.fixed_clim else None

        # Start fully "empty" at a very low dB so the first rows fade in from
        # the floor instead of flashing garbage.
        self.data = np.full((self.n_rows, freqs.size), -160.0, dtype=np.float32)

        self.fig, self.ax = plt.subplots(figsize=(11, 6))
        self.img = self.ax.imshow(
            self.data,
            aspect="auto",
            origin="upper",                       # row 0 (newest) at the top
            extent=[freqs[0], freqs[-1], self.n_rows * row_seconds, 0.0],
            cmap=cmap,
            interpolation="nearest",
            vmin=self._clim[0] if self.fixed_clim else -120.0,
            vmax=self._clim[1] if self.fixed_clim else -20.0,
        )
        self.ax.xaxis.set_major_formatter(EngFormatter("Hz"))
        self.ax.yaxis.set_major_formatter(EngFormatter("s"))
        self.ax.set_xlabel("RF frequency" if is_iq else "Frequency")
        self.ax.set_ylabel("Age (newest at top)")
        self.ax.set_title(title)
        self.fig.colorbar(self.img, ax=self.ax).set_label("[dB]")
        self.fig.tight_layout()

        self._status = self.ax.text(
            0.99, 0.015, "", transform=self.ax.transAxes, ha="right", va="bottom",
            color="white", fontsize=9,
            bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=2))

    def push_rows(self, rows: np.ndarray) -> None:
        """Scroll `rows` (oldest-first) into the top of the display."""
        if len(rows) == 0:
            return
        if len(rows) >= self.n_rows:
            # A burst larger than the window: keep only the newest screenful.
            self.data[:] = rows[-self.n_rows:][::-1]
        else:
            n = len(rows)
            self.data[n:] = self.data[:-n]        # push existing rows downward
            self.data[:n] = rows[::-1]            # newest row lands at index 0
        self._update_clim()

    def _update_clim(self) -> None:
        if self.fixed_clim:
            return
        # Ignore the not-yet-filled floor rows so startup doesn't skew the stats.
        live = self.data[self.data > -159.0]
        if live.size < 64:
            return
        # Same rule as the export renderers, on the visible window only.
        lo, hi = _clim(live, self._vmin, self._vmax, self.clim_mode)
        if self._clim is None:
            self._clim = (float(lo), float(hi))
        else:
            a = self._ADAPT_ALPHA
            self._clim = (float((1 - a) * self._clim[0] + a * lo),
                          float((1 - a) * self._clim[1] + a * hi))

    def draw(self, status: str = "") -> None:
        self.img.set_data(self.data)
        if self._clim is not None:
            self.img.set_clim(*self._clim)
        if status:
            self._status.set_text(status)
        self.fig.canvas.draw_idle()

    def save(self, path) -> None:
        self.fig.savefig(path, dpi=120)


def run_live(source, engine, *, seconds: float = 6.0, speed: float = 1.0,
             cmap: str = "magma", vmin=None, vmax=None, fps: int = 30,
             block_size: int = 8192, save_last=None,
             clim_mode: str = "floor") -> None:
    """Play a WAV through a scrolling waterfall, paced to real time.

    Pacing is wall-clock driven rather than "one block per animation frame":
    each timer tick we compute how much audio *should* have played by now
    (`speed` x elapsed) and pull exactly that many samples. That keeps the
    scroll rate correct and identical on a fast or slow machine, and makes
    --speed 4 mean genuinely 4x rather than "as fast as the GUI can go".
    """
    import matplotlib.pyplot as plt

    wf = Waterfall(engine.freqs, seconds=seconds, row_seconds=engine.seconds_per_row,
                   cmap=cmap, vmin=vmin, vmax=vmax, is_iq=source.is_iq,
                   clim_mode=clim_mode,
                   title=f"{source.path.name} — {source.sample_rate/1e3:.1f} kHz"
                         f"{' IQ' if source.is_iq else ''}")

    stream = source.blocks(block_size)
    state = {"consumed": 0, "t0": None, "done": False, "leftover": None}

    def pump(_frame=None):
        if state["done"]:
            return
        if state["t0"] is None:                   # start the clock on first draw
            state["t0"] = time.monotonic()
        elapsed = time.monotonic() - state["t0"]
        want = int(elapsed * speed * source.sample_rate) - state["consumed"]
        if want <= 0:
            return

        got = []
        while want > 0:
            chunk = state["leftover"]
            state["leftover"] = None
            if chunk is None:
                chunk = next(stream, None)
            if chunk is None:                     # end of file
                state["done"] = True
                break
            if len(chunk) > want:                 # keep the remainder for later
                state["leftover"] = chunk[want:]
                chunk = chunk[:want]
            got.append(chunk)
            want -= len(chunk)

        if got:
            samples = np.concatenate(got) if len(got) > 1 else got[0]
            state["consumed"] += len(samples)
            wf.push_rows(engine.push(samples))

        pos = state["consumed"] / source.sample_rate
        wf.draw(f"{pos:6.2f} / {source.duration:.2f} s"
                + (f"   x{speed:g}" if speed != 1.0 else "")
                + ("   [end]" if state["done"] else ""))

        if state["done"] and save_last:
            wf.save(save_last)
            print(f"Saved final frame to {save_last}")

    timer = wf.fig.canvas.new_timer(interval=int(1000 / max(fps, 1)))
    timer.add_callback(pump)
    timer.start()
    # Keep a reference alive: some backends garbage-collect a timer with no
    # remaining references and the animation silently stops after one frame.
    wf.fig._wav_spectrogram_timer = timer
    plt.show()
