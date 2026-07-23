"""App mode: a launcher screen, then the player, then back to the launcher.

No Tk. The launcher is drawn with matplotlib — the same toolkit the player and
the rest of this project already use, so there is exactly one GUI stack in the
process — and the file dialog is the *native* macOS one via `osascript`.

That is a deliberate retreat from tkinter: macOS ships a deprecated Tk 8.5 which
renders ttk windows blank on current systems, and matplotlib's Tk backend
creates its own Tk root per figure, which invalidates widgets belonging to any
other root. Both problems disappear once Tk is out of the picture.

Flow:

    launcher figure ──choose──▶ analyse (progress on the launcher)
        ▲                            │
        │                            ▼
        └──── Open… / Close ──── player figure
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

_STATE_FILE = Path.home() / ".wav_spectrogram.json"
_MAX_RECENT = 6

_NFFT_CHOICES = [256, 512, 1024, 2048, 4096, 8192]
_WINDOW_CHOICES = [2, 4, 6, 10, 20, 30]


# --------------------------------------------------------------------------
# persisted state (last folder + recent files)
# --------------------------------------------------------------------------

def _load_state() -> dict:
    try:
        state = json.loads(_STATE_FILE.read_text())
        return state if isinstance(state, dict) else {}
    except Exception:                       # missing/corrupt state is not an error
        return {}


def _save_state(state: dict) -> None:
    try:
        _STATE_FILE.write_text(json.dumps(state, indent=2))
    except OSError:
        pass                                # a read-only home is not worth failing over


def _last_dir(state: dict) -> str:
    d = state.get("last_dir", "")
    return d if d and Path(d).is_dir() else str(Path.home())


def _recent(state: dict) -> list[str]:
    return [p for p in state.get("recent", []) if Path(p).is_file()]


def _remember(state: dict, path: Path) -> None:
    recent = [str(path)] + [p for p in _recent(state) if p != str(path)]
    state["recent"] = recent[:_MAX_RECENT]
    state["last_dir"] = str(path.parent)
    _save_state(state)


# --------------------------------------------------------------------------
# native file dialog
# --------------------------------------------------------------------------

def pick_file(initial_dir: str | None = None) -> Path | None:
    """Native open dialog. Returns None if cancelled.

    macOS gets the real Finder dialog through osascript; anywhere else falls
    back to tkinter, which is fine on platforms with a current Tk.
    """
    if platform.system() == "Darwin":
        loc = ""
        if initial_dir and Path(initial_dir).is_dir():
            loc = f' default location POSIX file "{initial_dir}"'
        script = ('POSIX path of (choose file with prompt "Choose a WAV file"'
                  f' of type {{"wav"}}{loc})')
        try:
            out = subprocess.run(["osascript", "-e", script],
                                 capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if out.returncode != 0:             # user cancelled, or osascript failed
            return None
        path = out.stdout.strip()
        return Path(path) if path else None

    try:                                    # non-macOS fallback
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        try:
            path = filedialog.askopenfilename(
                title="Choose a WAV file", initialdir=initial_dir,
                filetypes=[("WAV audio", "*.wav *.WAV"), ("All files", "*.*")])
        finally:
            root.destroy()
        return Path(path) if path else None
    except Exception:
        return None


def pick_folder(prompt: str = "Choose a folder", initial_dir: str | None = None):
    """Native folder-chooser. Returns None if cancelled.

    Same split as pick_file: osascript on macOS so we never instantiate a Tk
    root beside matplotlib's, tkinter everywhere else.
    """
    if platform.system() == "Darwin":
        loc = ""
        if initial_dir and Path(initial_dir).is_dir():
            loc = f' default location POSIX file "{initial_dir}"'
        script = f'POSIX path of (choose folder with prompt "{prompt}"{loc})'
        try:
            out = subprocess.run(["osascript", "-e", script],
                                 capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if out.returncode != 0:             # cancelled
            return None
        path = out.stdout.strip()
        return Path(path) if path else None

    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        try:
            path = filedialog.askdirectory(title=prompt, initialdir=initial_dir)
        finally:
            root.destroy()
        return Path(path) if path else None
    except Exception:
        return None


def save_file(default_name: str, prompt: str = "Save as…",
              initial_dir: str | None = None):
    """Native save-as dialog. Returns None if cancelled.

    Used for the workbook export, which writes one named file rather than a
    folder of them.
    """
    if platform.system() == "Darwin":
        loc = ""
        if initial_dir and Path(initial_dir).is_dir():
            loc = f' default location POSIX file "{initial_dir}"'
        script = (f'POSIX path of (choose file name with prompt "{prompt}"'
                  f' default name "{default_name}"{loc})')
        try:
            out = subprocess.run(["osascript", "-e", script],
                                 capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if out.returncode != 0:             # cancelled
            return None
        path = out.stdout.strip()
        return Path(path) if path else None

    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        try:
            path = filedialog.asksaveasfilename(
                title=prompt, initialdir=initial_dir,
                initialfile=default_name, defaultextension=".xlsx",
                filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")])
        finally:
            root.destroy()
        return Path(path) if path else None
    except Exception:
        return None


# --------------------------------------------------------------------------
# launcher screen
# --------------------------------------------------------------------------

class Launcher:
    """Start screen: choose a file, set FFT size / window, see recent files.

    The analysis runs inside the button callback rather than after the figure
    closes, so the launcher itself can show progress and report a bad file
    without the screen going blank in between.
    """

    def __init__(self, args, state: dict):
        import matplotlib.pyplot as plt

        self.args = args
        self.state = state
        self.n_fft = int(args.nfft)
        self.window_s = float(args.seconds)
        self.result = None                  # (path, source, psd, freqs, row_s)

        self.fig = plt.figure(figsize=(9, 6.5))
        try:
            self.fig.canvas.manager.set_window_title("Spectrogram Player")
        except AttributeError:
            pass
        self.fig.patch.set_facecolor("#f6f6f7")

        self.fig.text(0.06, 0.90, "Spectrogram Player", fontsize=26, va="top")
        self.fig.text(0.06, 0.845,
                      "Choose a WAV file to see it as a spectrogram you can "
                      "play, pause, reverse and scrub.",
                      fontsize=11, color="#555", va="top")

        self._buttons = []                  # keep refs alive or callbacks die
        self._add_button([0.06, 0.70, 0.30, 0.08], "Choose WAV file…",
                         lambda e: self._choose(), fontsize=13)

        self.fig.text(0.42, 0.775, "FFT size", fontsize=10, color="#555")
        self.b_nfft = self._add_button([0.42, 0.70, 0.16, 0.055],
                                       str(self.n_fft), lambda e: self._cycle_nfft())
        self.fig.text(0.62, 0.775, "Window (s)", fontsize=10, color="#555")
        self.b_win = self._add_button([0.62, 0.70, 0.14, 0.055],
                                      f"{self.window_s:g}", lambda e: self._cycle_window())
        self.fig.text(0.79, 0.717,
                      "larger FFT =\nfiner frequency", fontsize=8, color="#888",
                      va="center")

        self.fig.text(0.06, 0.615, "Recent", fontsize=12, color="#333")
        recent = _recent(state)
        if recent:
            for i, item in enumerate(recent[:_MAX_RECENT]):
                p = Path(item)
                label = p.name if len(p.name) <= 46 else p.name[:43] + "…"
                self._add_button([0.06, 0.545 - i * 0.062, 0.70, 0.052], label,
                                 lambda e, q=p: self._open(q), fontsize=10)
        else:
            self.fig.text(0.06, 0.57, "Files you open will appear here.",
                          fontsize=10, color="#999", va="top")

        self.status = self.fig.text(0.06, 0.10, "", fontsize=11, color="#333")
        self._add_button([0.84, 0.05, 0.10, 0.06], "Quit", lambda e: self._quit())

    def _add_button(self, rect, label, cb, fontsize=11):
        from matplotlib.widgets import Button

        ax = self.fig.add_axes(rect)
        b = Button(ax, label, color="#ffffff", hovercolor="#e6e6e6")
        b.label.set_fontsize(fontsize)
        b.on_clicked(cb)
        self._buttons.append(b)
        return b

    # -- settings ----------------------------------------------------------

    def _cycle_nfft(self):
        i = (_NFFT_CHOICES.index(self.n_fft) + 1) % len(_NFFT_CHOICES) \
            if self.n_fft in _NFFT_CHOICES else 2
        self.n_fft = _NFFT_CHOICES[i]
        self.b_nfft.label.set_text(str(self.n_fft))
        self.fig.canvas.draw_idle()

    def _cycle_window(self):
        cur = int(self.window_s)
        i = (_WINDOW_CHOICES.index(cur) + 1) % len(_WINDOW_CHOICES) \
            if cur in _WINDOW_CHOICES else 2
        self.window_s = float(_WINDOW_CHOICES[i])
        self.b_win.label.set_text(f"{self.window_s:g}")
        self.fig.canvas.draw_idle()

    # -- actions -----------------------------------------------------------

    def _set_status(self, text, color="#333"):
        self.status.set_text(text)
        self.status.set_color(color)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _choose(self):
        path = pick_file(_last_dir(self.state))
        if path is not None:
            self._open(path)

    def _open(self, path: Path):
        import matplotlib.pyplot as plt

        from .stft import analyze
        from .wav_source import WavSource

        args = self.args
        hop = max(int(self.n_fft * (1.0 - args.overlap)), 1)
        try:
            source = WavSource(path, mode=args.mode, center_hz=args.center)
            self._set_status(f"Analysing {path.name}…")
            with source:
                psd, freqs, row_s = analyze(
                    source, n_fft=self.n_fft, hop=hop, avg=args.avg,
                    max_rows=args.player_max_rows,
                    progress=lambda f: self._set_status(
                        f"Analysing {path.name}…  {f*100:3.0f}%"))
        except Exception as exc:
            # Stay on the launcher and say what went wrong, rather than exiting.
            self._set_status(f"Could not open {path.name}: {exc}", color="#b00")
            return

        _remember(self.state, path)
        self.result = (path, source, psd, freqs, row_s, self.window_s)
        plt.close(self.fig)

    def _quit(self):
        import matplotlib.pyplot as plt
        self.result = None
        plt.close(self.fig)

    def show(self):
        """Block until a file is chosen or the launcher is closed."""
        import matplotlib.pyplot as plt
        plt.show()
        return self.result


# --------------------------------------------------------------------------
# app loop
# --------------------------------------------------------------------------

def run_app(args, hop: int) -> int:
    """`hop` is ignored: it is recomputed per file from the launcher's setting."""
    import matplotlib
    if matplotlib.get_backend().lower() in ("agg", "template"):
        print("No interactive display available — use --export instead.")
        return 1

    from .player import SpectrogramPlayer

    state = _load_state()
    while True:
        chosen = Launcher(args, state).show()
        if chosen is None:
            return 0
        path, source, psd, freqs, row_s, window_s = chosen
        print(f"{path.name}: {psd.shape[0]} rows x {psd.shape[1]} bins "
              f"({psd.nbytes/1e6:.0f} MB, {1/row_s:.0f} rows/s)")

        player = SpectrogramPlayer(
            psd, freqs, row_s, window_s=window_s, cmap=args.cmap,
            vmin=args.vmin, vmax=args.vmax, clim_mode=args.clim_mode,
            is_iq=source.is_iq, loop=args.loop, speed=args.speed, show_open=True,
            view=args.view, traces=args.traces, trace_bins=args.trace_bins,
            csv_stem=path.stem, csv_snr=args.snr,
            csv_min_duration=args.min_duration or 0.05,
            title=f"{path.name} — {source.sample_rate/1e3:.1f} kHz"
                  f"{' IQ' if source.is_iq else ''}")
        player.show(fps=args.fps)
        # Whether the user pressed Open…, Close, or shut the window, the next
        # stop is the launcher — there is no way to get stranded with no window.
