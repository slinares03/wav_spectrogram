"""Spectrogram player — a video-player transport over a WAV's spectrogram.

Why this exists separately from waterfall.py: a streaming waterfall can only go
forward, because it consumes the file through a one-shot generator. Seeking,
reverse and scrubbing all need random access, so the player precomputes the
whole spectrogram up front (the same trade the labeling tool makes with its
on-disk cache) and then playback is just a moving window into that matrix.
Everything after the precompute is O(window), so scrubbing is instant.

Layout, top to bottom:

    main view   frequency across, time downward — the newest row is at the
                bottom edge and older audio slides up and off the top, so the
                image builds the way a live spectrogram prints.
    overview    a vertical strip beside the main view: the whole file at a
                glance, sharing its (downward) time axis, and the scrub bar —
                click or drag anywhere on it to seek.
    transport   |< << play >> >| buttons, speed, and a status readout.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from .export import clim as _clim

# Keys matplotlib binds by default that we want for transport.
_CONFLICTING_KEYMAPS = ("keymap.save", "keymap.yscale", "keymap.xscale",
                        "keymap.home", "keymap.back", "keymap.forward",
                        "keymap.pan", "keymap.zoom", "keymap.grid",
                        "keymap.grid_minor", "keymap.fullscreen")

_SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

HELP = ("space play/pause   J/K/L reverse/pause/forward   R flip direction   "
        "←→ step 1s (shift 5s)   ↑↓ speed   , . contrast   "
        "home/end   E export Excel   click or drag the strip to seek")

HELP_3D = ("3-D axes:   7/8/9 swap XY/YZ/XZ   "
           "x/y/z flip an axis   0 reset")

# Keyboard shortcuts for the 3-D axis controls, mapped to the same action names
# the side buttons use so both paths share one handler.
_KEY_TO_3D = {"7": "swap_xy", "8": "swap_yz", "9": "swap_xz",
              "x": "flip_x", "y": "flip_y", "z": "flip_z", "0": "reset"}


class SpectrogramPlayer:
    """Transport UI over a precomputed spectrogram matrix.

    Args:
        psd:          (time, freq) dB matrix.
        freqs:        frequency axis in Hz.
        row_seconds:  seconds of audio per row.
        window_s:     seconds of spectrogram visible in the main view.
        title:        window title.
        loop:         restart at the far end instead of stopping.
    """

    _LEAD = 1.0               # all of the window is already-played audio

    def __init__(self, psd, freqs, row_seconds, window_s=6.0, cmap="magma",
                 vmin=None, vmax=None, clim_mode="floor", is_iq=False,
                 title="", loop=False, speed=1.0, fig=None,
                 show_open=False, on_open=None, on_quit=None,
                 view="2d", traces=48, trace_bins=192,
                 csv_stem="spectrogram", csv_snr=12.0,
                 csv_min_duration=0.05):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import EngFormatter
        from matplotlib.widgets import Button

        self.psd = psd
        self.freqs = freqs
        self.row_seconds = row_seconds
        self.n_rows = psd.shape[0]
        # Defaults for the Export CSV button. min_duration matches what the
        # CLI docs recommend rather than the CLI's raw default of 0: a UI
        # button should hand back the usable list, not the noisy one.
        self.csv_stem = csv_stem
        self.csv_snr = float(csv_snr)
        self.csv_min_duration = float(csv_min_duration)
        self.duration = self.n_rows * row_seconds
        self.loop = loop
        # Kept for the still-image export button (render_preview/render_surface).
        self.cmap = cmap
        self.is_iq = is_iq
        self.title = title
        self.clim_mode = clim_mode
        self.traces = int(traces)
        self.trace_bins = int(trace_bins)

        self.n_win = max(int(round(window_s / row_seconds)), 8)
        self.pos = 0.0                                  # playhead, in rows
        self.playing = True
        self.direction = 1
        self.speed = float(speed)
        self._scrubbing = False
        self._last_tick = None
        self._closed = False
        self._timer = None

        # Contrast is computed once over the whole file, so it never pulses as
        # loud content scrolls past — the main advantage of precomputing.
        self.vmin, self.vmax = _clim(psd, vmin, vmax, clim_mode)

        for key in _CONFLICTING_KEYMAPS:
            plt.rcParams[key] = []

        # A caller (app.py) can pass a Figure it has already embedded in its own
        # window. Matplotlib's Tk backend creates a fresh Tk root per figure,
        # and a second root in one process invalidates the first one's widgets —
        # so in app mode the app owns the single root and we draw into its
        # canvas rather than letting pyplot open a window of its own.
        self._standalone = fig is None
        self.fig = plt.figure(figsize=(12, 7.5)) if self._standalone else fig
        if self._standalone:
            try:
                self.fig.canvas.manager.set_window_title(title or "wav_spectrogram")
            except AttributeError:
                pass                       # no window manager (e.g. Agg)

        # --- main view ---------------------------------------------------
        self.view = view
        self.view3d = None
        self.img = None
        self.playhead = None
        # Blitting is only used by the 3-D view, where a full redraw costs
        # ~250 ms; the 2-D imshow path is already fast enough with draw_idle.
        self._blit_bg = None

        if view == "3d":
            from .surface3d import RibbonView
            # Inset from both edges so the axis-control columns have gutters to
            # sit in — flips on the left of the plot, swaps/reset on the right.
            self.ax = self.fig.add_axes([0.11, 0.20, 0.57, 0.76],
                                        projection="3d")
            self.view3d = RibbonView(
                self.ax, freqs, window_s=self.n_win * row_seconds,
                vmin=self.vmin, vmax=self.vmax, cmap=cmap,
                n_traces=traces, n_bins=trace_bins, is_iq=is_iq, title=title)
            # Any full redraw — first draw, resize, or a camera drag — makes the
            # cached background stale, so recapture it whenever one happens.
            self.fig.canvas.mpl_connect("draw_event", self._on_draw)
            self._build_axis_controls(Button)
        else:
            # Frequency across, time *downward*: the newest row is at the bottom
            # edge and older audio slides up and off the top. The y axis is
            # inverted rather than the data flipped, so every time value in the
            # code stays a real file timestamp.
            self.ax = self.fig.add_axes([0.07, 0.22, 0.66, 0.72])
            self.img = self.ax.imshow(
                self._window_data(), aspect="auto", origin="lower",
                extent=self._window_extent(), cmap=cmap,
                vmin=self.vmin, vmax=self.vmax, interpolation="nearest")
            self.ax.set_xlabel("RF frequency" if is_iq else "Frequency")
            self.ax.set_ylabel("Time")
            self.ax.xaxis.set_major_formatter(EngFormatter("Hz"))
            self.ax.yaxis.set_major_formatter(EngFormatter("s"))
            self.ax.set_title(title, fontsize=10)
            self.ax.invert_yaxis()
            self.playhead = self.ax.axhline(0.0, color="white", lw=1.2, alpha=0.9)

        # --- overview / scrub bar ----------------------------------------
        # Vertical, beside the main view, so both share the same time axis.
        self.ax_ov = self.fig.add_axes([0.78, 0.22, 0.15, 0.72])
        self.ax_ov.imshow(self._overview_data(), aspect="auto", origin="lower",
                          extent=[freqs[0], freqs[-1], 0, self.duration],
                          cmap=cmap, vmin=self.vmin, vmax=self.vmax,
                          interpolation="nearest")
        self.ax_ov.set_xticks([])
        self.ax_ov.yaxis.tick_right()
        self.ax_ov.yaxis.set_major_formatter(EngFormatter("s"))
        self.ax_ov.set_title("whole file\nclick or drag to seek", fontsize=8,
                             color="#555")
        self.ax_ov.invert_yaxis()          # start of file at the top, like the main view
        self.ov_head = self.ax_ov.axhline(0.0, color="white", lw=1.5)
        # Shade the span currently visible in the main view.
        self.ov_span = self.ax_ov.axhspan(0, 0, color="white", alpha=0.22, lw=0)

        # --- transport ----------------------------------------------------
        specs = [("|<", self._go_start), ("<<", lambda e: self._nudge(-1.0)),
                 ("<", self._play_reverse), ("||", self._pause),
                 (">", self._play_forward), (">>", lambda e: self._nudge(1.0)),
                 (">|", self._go_end)]
        self.buttons = []                 # keep refs alive or callbacks die
        x = 0.03                          # tighter pitch so the row clears the
        for label, cb in specs:           # export buttons further right
            ax_b = self.fig.add_axes([x, 0.085, 0.046, 0.055])
            b = Button(ax_b, label, hovercolor="0.85")
            b.label.set_fontsize(11)
            b.on_clicked(cb)
            self.buttons.append(b)
            x += 0.050

        ax_slow = self.fig.add_axes([x + 0.012, 0.085, 0.032, 0.055])
        ax_fast = self.fig.add_axes([x + 0.078, 0.085, 0.032, 0.055])
        self.b_slow = Button(ax_slow, "-", hovercolor="0.85")
        self.b_fast = Button(ax_fast, "+", hovercolor="0.85")
        self.b_slow.on_clicked(lambda e: self._change_speed(-1))
        self.b_fast.on_clicked(lambda e: self._change_speed(+1))
        self.speed_text = self.fig.text(x + 0.055, 0.112, "", ha="center",
                                        va="center", fontsize=10)

        # --- export -------------------------------------------------------
        # The player already holds the whole file's spectrogram in memory, so
        # exporting is pure analysis/rendering with nothing to re-read off disk.
        ax_csv = self.fig.add_axes([0.52, 0.085, 0.10, 0.055])
        self.b_csv = Button(ax_csv, "Export Excel", hovercolor="0.85")
        self.b_csv.label.set_fontsize(8)
        self.b_csv.on_clicked(self._on_export_csv)
        ax_png = self.fig.add_axes([0.625, 0.085, 0.10, 0.055])
        self.b_png = Button(ax_png, "Export PNGs", hovercolor="0.85")
        self.b_png.label.set_fontsize(8)
        self.b_png.on_clicked(self._on_export_png)
        # Its own line, so a result message is not overwritten by the transport
        # readout on the very next timer tick. Shared by both export buttons.
        self.csv_status = self.fig.text(0.52, 0.055, "", ha="left", va="center",
                                        fontsize=7.5, color="0.35")

        # In app mode the player hands control back to the launcher rather than
        # ending the program, so closing this window is not the only exit.
        self.reopen = False
        self.on_open, self.on_quit = on_open, on_quit
        if show_open or on_open is not None:
            ax_open = self.fig.add_axes([0.80, 0.085, 0.085, 0.055])
            self.b_open = Button(ax_open, "Open…", hovercolor="0.85")
            self.b_open.on_clicked(self._on_open)
            ax_quit = self.fig.add_axes([0.895, 0.085, 0.075, 0.055])
            self.b_quit = Button(ax_quit, "Close", hovercolor="0.85")
            self.b_quit.on_clicked(self._on_quit)

        # Above the button row rather than beside it: Export CSV now occupies
        # the gap this used to sit in, and a long "|> 123.45 / 456.78 s" readout
        # is wide enough to have run underneath it.
        self.status = self.fig.text(0.97, 0.170, "", ha="right", va="center",
                                    fontsize=10, family="monospace")
        self.fig.text(0.5, 0.028, HELP, fontsize=7.5, color="0.45",
                      ha="center", va="center")
        # 3-D remap keys only mean anything in the ribbon view, so their legend
        # is shown there and nowhere else.
        if self.view3d is not None:
            self.fig.text(0.5, 0.006, HELP_3D, fontsize=7.5, color="#3060a0",
                          ha="center", va="center")
            # The 3-D path blits: it restores a cached background and redraws
            # just these artists on top. Mark them animated so the full draw
            # that captures the background leaves them out — otherwise each new
            # value is painted over the stale one baked into the cache, and the
            # time readout in particular smears into an unreadable overlap.
            for art in (self.status, self.speed_text, self.csv_status,
                        self.ov_head, self.ov_span, self.view3d.collection):
                art.set_animated(True)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    # -- data slicing ------------------------------------------------------

    def _overview_data(self) -> np.ndarray:
        """Whole file squeezed to ~1500 columns for the scrub bar."""
        target = 1500
        p = self.psd
        if len(p) > target:
            g = len(p) // target
            p = p[: g * target].reshape(target, g, -1).max(axis=1)
        return p                                     # (time, freq)

    def _window_extent(self):
        i0 = self.pos - self.n_win * self._LEAD
        return [self.freqs[0], self.freqs[-1],
                i0 * self.row_seconds, (i0 + self.n_win) * self.row_seconds]

    def _window_data(self) -> np.ndarray:
        """Visible slice as (time, freq), padded past the file's ends."""
        i0 = int(round(self.pos - self.n_win * self._LEAD))
        i1 = i0 + self.n_win
        out = np.full((self.n_win, self.psd.shape[1]), self.vmin, dtype=np.float32)
        a, b = max(i0, 0), min(i1, self.n_rows)
        if a < b:
            out[a - i0: b - i0] = self.psd[a:b]
        return out

    def _on_open(self, _e=None):
        """Ask the app for a new file, then tear this window down."""
        self.reopen = True
        if self.on_open is not None:      # embedded: the app owns the window
            self.stop()
            self.on_open()
        else:                             # standalone: closing ends show()
            self._close()

    def _set_csv_status(self, msg: str, color: str = "0.35") -> None:
        self.csv_status.set_text(msg)
        self.csv_status.set_color(color)
        # Paint it now: the analysis below blocks the GUI thread, so a message
        # queued with draw_idle would not appear until after the work it is
        # meant to announce has already finished.
        if self.view3d is None:
            self.fig.canvas.draw()
            return
        # In 3-D the status text is an animated artist, excluded from the full
        # draw; refresh the background if needed, then blit it in so the message
        # actually shows instead of being skipped.
        if self._blit_bg is None:
            self.fig.canvas.draw()
        self._blit_frame()

    def _on_export_csv(self, _e=None):
        """Analyse the whole file and save it as a spreadsheet."""
        from .app import save_file
        from .detect import build_sheets, write_csv, write_xlsx, xlsx_available

        was_playing = self.playing
        self.playing = False                  # don't scroll under the dialog
        try:
            as_xlsx = xlsx_available()
            ext = "xlsx" if as_xlsx else "csv"
            target = save_file(f"{self.csv_stem}_analysis.{ext}",
                               "Export spectrogram analysis…")
            if target is None:                # cancelled
                self._set_csv_status("")
                return

            self._set_csv_status("analysing…")
            times = np.arange(self.n_rows) * self.row_seconds
            sheets = build_sheets(
                self.psd, self.freqs, times, snr_db=self.csv_snr,
                min_duration_s=self.csv_min_duration)

            if as_xlsx:
                info = {
                    "source file": self.csv_stem,
                    "duration (s)": round(self.duration, 3),
                    "time resolution (ms)": round(self.row_seconds * 1000, 2),
                    "frequency resolution (Hz)": round(
                        float(self.freqs[1] - self.freqs[0]), 3),
                    "snr threshold (dB)": self.csv_snr,
                    "min event duration (s)": self.csv_min_duration,
                }
                counts = write_xlsx(target, sheets, info=info)
                written = target.with_suffix(".xlsx")
            else:
                # No openpyxl: fall back to a CSV per table beside the chosen
                # name, rather than refusing to export at all.
                counts = {}
                for title, rows, columns in sheets:
                    path = target.with_name(
                        f"{target.stem}_{title.lower()}.csv")
                    counts[title] = write_csv(rows, path, columns=columns)
                written = target.parent

            summary = ", ".join(f"{n} {t.lower()}" for t, n in counts.items())
            self._set_csv_status(f"saved {summary} → {written.name}",
                                 color="#1a7f37")
            print(f"export: {written}  ({summary})")
        except Exception as exc:              # a failed export must not kill playback
            self._set_csv_status(f"export failed: {exc}", color="#b3261e")
            print(f"CSV export failed: {exc}", file=sys.stderr)
        finally:
            self.playing = was_playing
            self._last_tick = None            # don't jump by the time spent here
            self._blit_bg = None              # message changed the static layer

    def _on_export_png(self, _e=None):
        """Render the whole file to two stills: the 2-D spectrogram and the
        3-D amplitude surface, sharing one chosen base name."""
        from .app import save_file
        from .export import render_preview, render_surface

        was_playing = self.playing
        self.playing = False                   # don't scroll under the dialog
        try:
            target = save_file(f"{self.csv_stem}_spectrogram.png",
                               "Export spectrogram images…")
            if target is None:                 # cancelled
                self._set_csv_status("")
                return

            self._set_csv_status("rendering…")
            times = np.arange(self.n_rows) * self.row_seconds
            title = self.title or self.csv_stem
            # Both stills land beside the chosen name, with the 2-D image taking
            # it verbatim and the 3-D surface a matching "_surface" sibling.
            stem = target.with_suffix("").name
            if stem.endswith("_spectrogram"):
                stem = stem[: -len("_spectrogram")]
            flat = target.with_name(f"{stem}_spectrogram.png")
            surf = target.with_name(f"{stem}_surface.png")

            render_preview(self.psd, self.freqs, times, flat, title=title,
                           cmap=self.cmap, vmin=self.vmin, vmax=self.vmax,
                           is_iq=self.is_iq, clim_mode=self.clim_mode)
            # In 3-D, render the still through the live view so the export
            # mirrors whatever axis layout is on screen (swaps, flips, camera);
            # in 2-D there is no such layout, so fall back to the default surface.
            if self.view3d is not None:
                self._render_surface_still(surf, title)
            else:
                render_surface(self.psd, self.freqs, times, surf, title=title,
                               cmap=self.cmap, vmin=self.vmin, vmax=self.vmax,
                               is_iq=self.is_iq, clim_mode=self.clim_mode)

            self._set_csv_status(
                f"saved {flat.name} + {surf.name}", color="#1a7f37")
            print(f"export: {flat}\n        {surf}")
        except Exception as exc:               # a failed export must not kill playback
            self._set_csv_status(f"export failed: {exc}", color="#b3261e")
            print(f"PNG export failed: {exc}", file=sys.stderr)
        finally:
            self.playing = was_playing
            self._last_tick = None             # don't jump by the time spent here
            self._blit_bg = None               # message changed the static layer

    def _render_surface_still(self, out_path, title: str) -> None:
        """Save a 3-D still that matches the live ribbon view's current layout.

        A fresh RibbonView is drawn onto an off-screen Agg figure, then given
        the live view's axis mapping, flips and camera angle, so a flipped or
        swapped axis on screen appears the same way in the exported image. The
        whole file is fed as one window (more traces than playback) for detail.
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from .surface3d import RibbonView

        fig = Figure(figsize=(12, 7))
        FigureCanvasAgg(fig)                    # explicit Agg: never touch the GUI
        ax = fig.add_subplot(111, projection="3d")
        n_traces = max(self.traces, min(160, self.n_rows))
        rv = RibbonView(ax, self.freqs, window_s=self.duration,
                        vmin=self.vmin, vmax=self.vmax, cmap=self.cmap,
                        n_traces=n_traces, n_bins=self.trace_bins,
                        is_iq=self.is_iq, title=title)
        # Mirror the on-screen configuration, then rebuild geometry/labels.
        rv.src = dict(self.view3d.src)
        rv.flip = dict(self.view3d.flip)
        rv._apply_layout()
        ax.view_init(elev=self.view3d.ax.elev, azim=self.view3d.ax.azim)
        rv.set_window(self.psd)
        fig.savefig(out_path, dpi=120)

    def _on_quit(self, _e=None):
        if self.on_quit is not None:
            self.stop()
            self.on_quit()
        else:
            self._close()

    def stop(self):
        """Halt playback and the redraw timer. Safe to call more than once."""
        self.playing = False
        self._closed = True
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:
                pass
            self._timer = None

    def _close(self):
        import matplotlib.pyplot as plt
        self.stop()
        if self._standalone:
            plt.close(self.fig)

    def _set_span(self, y0: float, y1: float):
        """Move the overview's visible-range shading.

        axhspan returns a Rectangle on some matplotlib versions and a Polygon on
        others, and the two take different geometry APIs — handle both.
        """
        if hasattr(self.ov_span, "set_y"):                 # Rectangle
            self.ov_span.set_y(y0)
            self.ov_span.set_height(y1 - y0)
        else:                                              # Polygon
            self.ov_span.set_xy([[0, y0], [0, y1], [1, y1], [1, y0], [0, y0]])

    # -- transport ---------------------------------------------------------

    def _play_forward(self, _e=None):
        self.direction, self.playing = 1, True

    def _play_reverse(self, _e=None):
        self.direction, self.playing = -1, True

    def _pause(self, _e=None):
        self.playing = False

    def _toggle(self):
        self.playing = not self.playing

    def _go_start(self, _e=None):
        self.pos, self.playing = 0.0, False

    def _go_end(self, _e=None):
        self.pos, self.playing = float(self.n_rows), False

    def _nudge(self, seconds: float):
        self.pos = float(np.clip(self.pos + seconds / self.row_seconds,
                                 0, self.n_rows))

    def _change_speed(self, step: int):
        # Snap to the nearest preset, then move along the list.
        i = int(np.argmin([abs(s - self.speed) for s in _SPEEDS]))
        self.speed = _SPEEDS[int(np.clip(i + step, 0, len(_SPEEDS) - 1))]

    def _nudge_contrast(self, db: float):
        self.vmin = min(self.vmin + db, self.vmax - 3.0)
        if self.view3d is not None:
            # Moves the z limits, so the axes change: force a full redraw to
            # rebuild the blit background rather than blitting onto a stale one.
            self.view3d.set_clim(self.vmin, self.vmax)
            self._blit_bg = None
        else:
            self.img.set_clim(self.vmin, self.vmax)

    def _seek_to_time(self, t_s: float):
        self.pos = float(np.clip(t_s / self.row_seconds, 0, self.n_rows))

    # -- events ------------------------------------------------------------

    def _on_key(self, event):
        k = event.key
        if k == " ":
            self._toggle()
        elif k in ("l", "L"):
            self._play_forward()
        elif k in ("j", "J"):
            self._play_reverse()
        elif k in ("k", "K"):
            self._pause()
        elif k in ("r", "R"):
            self.direction *= -1
        elif k == "right":
            self._nudge(1.0)
        elif k == "left":
            self._nudge(-1.0)
        elif k == "shift+right":
            self._nudge(5.0)
        elif k == "shift+left":
            self._nudge(-5.0)
        elif k == "up":
            self._change_speed(+1)
        elif k == "down":
            self._change_speed(-1)
        elif k == "home":
            self._go_start()
        elif k == "end":
            self._go_end()
        elif k in ("e", "E"):
            self._on_export_csv()
            return                     # the handler already redrew
        elif k == ",":
            self._nudge_contrast(-2.0)
        elif k == ".":
            self._nudge_contrast(+2.0)
        elif self.view3d is not None and k in _KEY_TO_3D:
            self._do_3d(_KEY_TO_3D[k])
            return                     # the handler already redrew
        else:
            return
        self._render()

    # -- 3-D axis controls -------------------------------------------------

    # Buttons down each side of the ribbon plot: flips on the left, swaps and
    # reset on the right. (label, action) pairs, laid out top-to-bottom.
    _FLIP_CONTROLS = (("Flip X", "flip_x"), ("Flip Y", "flip_y"),
                      ("Flip Z", "flip_z"))
    _SWAP_CONTROLS = (("Swap X/Y", "swap_xy"), ("Swap Y/Z", "swap_yz"),
                      ("Swap X/Z", "swap_xz"), ("Reset", "reset"))

    def _build_axis_controls(self, Button) -> None:
        """Create the side button columns for the 3-D view.

        Kept out of __init__'s main flow because it only runs in 3-D; the button
        refs live on self so their callbacks are not garbage-collected.
        """
        self.axis_buttons = []
        W, H, PITCH = 0.072, 0.045, 0.058       # compact, so all four swaps fit

        def column(specs, x, top):
            for i, (label, action) in enumerate(specs):
                ax_b = self.fig.add_axes([x, top - i * PITCH, W, H])
                b = Button(ax_b, label, hovercolor="0.85")
                b.label.set_fontsize(7.5)
                # Default arg binds this iteration's action, not the last one.
                b.on_clicked(lambda _e, a=action: self._do_3d(a))
                self.axis_buttons.append(b)

        left_x, right_x = 0.018, 0.700
        # Small captions above each column so the grouping reads at a glance.
        self.fig.text(left_x + W / 2, 0.735, "flip", ha="center",
                      fontsize=8, color="0.4")
        self.fig.text(right_x + W / 2, 0.735, "swap", ha="center",
                      fontsize=8, color="0.4")
        column(self._FLIP_CONTROLS, left_x, 0.68)
        column(self._SWAP_CONTROLS, right_x, 0.68)

    def _do_3d(self, action: str) -> None:
        """Apply a 3-D axis remap/flip/reset, then redraw.

        Both the side buttons and the 7/8/9/x/y/z/0 keys route through here. Any
        of these moves the axis limits and labels, so the blit background is
        invalidated and a full redraw taken.
        """
        v = self.view3d
        actions = {
            "swap_xy": lambda: v.swap_axes("x", "y"),
            "swap_yz": lambda: v.swap_axes("y", "z"),
            "swap_xz": lambda: v.swap_axes("x", "z"),
            "flip_x": lambda: v.flip_axis("x"),
            "flip_y": lambda: v.flip_axis("y"),
            "flip_z": lambda: v.flip_axis("z"),
            "reset": v.reset_axes,
        }
        actions[action]()
        self.csv_status.set_text(f"3-D axes  {v.describe()}")
        self.csv_status.set_color("0.35")
        self._blit_bg = None           # limits/labels moved: recapture on redraw
        self._render()

    def _on_press(self, event):
        if event.inaxes is self.ax_ov and event.ydata is not None:
            self._scrubbing = True
            self._seek_to_time(event.ydata)
            self._render()
        elif (self.view3d is None and event.inaxes is self.ax
                and event.ydata is not None):
            self._seek_to_time(event.ydata)      # click the main view to seek
            self._render()
        # In 3-D the main view has no time axis to click (X is time *within the
        # window*) and a drag there belongs to the camera, so seeking is left to
        # the scrub bar and the transport keys.

    def _on_motion(self, event):
        if self._scrubbing and event.inaxes is self.ax_ov and event.ydata is not None:
            self._seek_to_time(event.ydata)
            self._render()

    def _on_release(self, _event):
        self._scrubbing = False

    # -- loop --------------------------------------------------------------

    def _tick(self):
        if self._closed:                   # a queued tick can outlive the window
            return
        now = time.monotonic()
        dt = 0.0 if self._last_tick is None else now - self._last_tick
        self._last_tick = now

        if self.playing and not self._scrubbing and dt > 0:
            # Wall-clock advance, so speed is honest regardless of frame rate.
            self.pos += self.direction * self.speed * dt / self.row_seconds
            if self.pos >= self.n_rows:
                self.pos = 0.0 if self.loop else float(self.n_rows)
                self.playing = self.loop
            elif self.pos <= 0.0:
                self.pos = float(self.n_rows) if self.loop else 0.0
                self.playing = self.loop
        self._render()

    def _on_draw(self, _event=None):
        """Cache the freshly rendered figure as the blit background."""
        if self.view3d is not None:
            self._blit_bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _render(self):
        ext = self._window_extent()
        t = self.pos * self.row_seconds

        if self.view3d is not None:
            self.view3d.set_window(self._window_data())
        else:
            self.img.set_data(self._window_data())
            self.img.set_extent(ext)
            self.ax.set_ylim(ext[3], ext[2])  # inverted: newest (ext[3]) at the bottom
            self.playhead.set_ydata([t, t])

        self.ov_head.set_ydata([t, t])
        self._set_span(ext[2], ext[3])

        arrow = "|>" if self.direction > 0 else "<|"
        state = arrow if self.playing else "||"
        self.speed_text.set_text(f"x{self.speed:g}")
        self.status.set_text(f"{state}  {t:6.2f} / {self.duration:6.2f} s")

        if self.view3d is None:
            self.fig.canvas.draw_idle()
            return

        if self._blit_bg is None:
            # No valid background yet (first frame, resize, camera move): take
            # the slow path once, which repopulates it via the draw_event hook.
            self.fig.canvas.draw()
            return
        self._blit_frame()

    def _blit_frame(self) -> None:
        """Restore the cached background and redraw only the animated overlay.

        Requires a valid self._blit_bg. Shared by the per-frame render and the
        export-status update so the two cannot drift in which artists they paint.
        """
        self.fig.canvas.restore_region(self._blit_bg)
        # draw_artist bypasses the axes' own draw, which is what normally runs
        # the 3-D projection — do it by hand or the ribbons never update.
        self.view3d.project()
        for artist in (self.view3d.collection, self.ov_head, self.ov_span,
                       self.speed_text, self.status, self.csv_status):
            self.ax.figure.draw_artist(artist)
        self.fig.canvas.blit(self.fig.bbox)

    def start(self, fps: int = 30) -> None:
        """Begin the redraw/playback timer. Use when the canvas is embedded."""
        self._timer = self.fig.canvas.new_timer(interval=int(1000 / max(fps, 1)))
        self._timer.add_callback(self._tick)
        self._timer.start()
        self.fig._player_timer = self._timer   # some backends GC a loose timer
        self._render()

    def show(self, fps: int = 30) -> bool:
        """Standalone: block until the window closes (CLI single-file mode)."""
        import matplotlib.pyplot as plt
        self.fig.canvas.mpl_connect("close_event", lambda e: self.stop())
        self.start(fps)
        plt.show()
        self.stop()
        return self.reopen
