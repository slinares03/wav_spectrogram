# wav_spectrogram

A spectrogram **player** for WAV files — transport controls, seek, reverse,
scrub — plus a streaming waterfall and still-PNG export.

Unlike `vita49_pipeline/capture_spectrogram.py`, which snapshots a live UDP
stream to one image, this plays a *file* back: a WAV has its own timeline, so
the natural interface is a video player over that timeline rather than a single
picture. The still export is kept for feeding the ML/labeling workflow.

Run it with no arguments and it behaves like a small desktop app: a launcher
window opens with a "Choose WAV file…" button, FFT/window settings and your
recent files. Pick a file — progress shows on the launcher — and the player
opens. "Open…" or "Close" in the player returns to the launcher; "Quit" there
exits.

```bash
./wav-player --view 3d          # macOS / Linux, from the project root
.\wav-player --view 3d          # Windows (PowerShell or cmd)
```

On any platform you can also run it as a module — `python -m wav_spectrogram`
(from the parent of this folder) — which is exactly what the launchers do.

Five modes:

| mode | invocation | when |
|---|---|---|
| app | no arguments | launcher window: choose a file, play, choose another |
| player | a filename | skip the picker and play that file |
| stream | `--stream` | forward-only waterfall; starts instantly, unbounded length |
| export | `--export DIR` | headless PNGs for the ML/labeling pipeline |
| analysis | `--xlsx DIR` | scan the file and log what was detected, as an Excel workbook |

Handles both kinds of WAV in this project:

| file | `--mode` | spectrum |
|---|---|---|
| ordinary audio (mono/stereo) | `audio` | one-sided, 0 – fs/2 |
| SDR capture, L=I / R=Q | `iq` | two-sided, centred on `--center` |

`--mode auto` (default) only reads a 2-channel file as IQ when `--center` is
given or the filename contains `iq`/`baseband`/`complex` — so ordinary stereo is
never misread as complex baseband. Be explicit when it matters.

## Install

```bash
pip install -r wav_spectrogram/requirements.txt
```

`numpy` + `matplotlib` are required; `pillow` only for `--yolo`. `soundfile` is
optional — without it the stdlib decoder handles PCM 8/16/24/32-bit WAV, with it
you also get float and WAVEX files.

## Use

The `./wav-player` / `.\wav-player` launcher and `python -m wav_spectrogram` are
interchangeable — use whichever you prefer. The examples below use the module
form because it reads the same on every platform.

```bash
# app — file picker, then play in the 3-D view; loops until you quit
./wav-player --view 3d                    # macOS / Linux
.\wav-player --view 3d                    # Windows
python -m wav_spectrogram --view 3d       # any platform

# live 3-D ribbon view straight to a file — the reconfigurable amplitude relief
./wav-player recording.wav --view 3d                 # macOS / Linux
.\wav-player recording.wav --view 3d                 # Windows
python -m wav_spectrogram recording.wav --view 3d    # any platform

# 3-D, 10 s visible at once, finer frequency resolution, looping
python -m wav_spectrogram recording.wav --view 3d --seconds 10 --nfft 4096 --loop

# 3-D of an SDR capture tuned to 433.92 MHz
python -m wav_spectrogram capture.wav --view 3d --mode iq --center 433.92e6

# plain 2-D player (the default) straight to a file
python -m wav_spectrogram recording.wav

# forward-only waterfall, no precompute wait
python -m wav_spectrogram recording.wav --stream

# no window: whole file to PNG, plus a YOLO-ready 640x640 tile
python -m wav_spectrogram recording.wav --export out/ --yolo

# 3-D surface: time across, frequency into the page, amplitude as height
python -m wav_spectrogram recording.wav --export out/ --surface
```

### 3-D surface view

`--export --surface` writes `<name>_surface.png` alongside the flat preview.
Where the flat images encode amplitude as colour only, here amplitude is the
vertical axis, so peaks stand out of the noise floor instead of blending into
it. Colour still tracks height, using the same dB limits (`--vmin`/`--vmax`/
`--clim-mode`) as the 2-D images so the two read consistently.

The grid is block-averaged down to roughly 400 x 200 cells — a surface draws one
quad per cell, and the raw matrix is both unreadable and slow. Low frequencies
are placed at the far edge, since on typical audio their tall ridge would
otherwise hide everything behind it. Change the camera with
`--surface-view ELEV AZIM` (default `35 -60`).

### Spreadsheet analysis (`--xlsx`)

Scans the whole file and writes one `.xlsx` workbook — no window opens. Opens
by double-click in Excel and via File → Import in Google Sheets.

```bash
./wav-player recording.wav --xlsx logs/                    # one workbook
./wav-player recording.wav --xlsx logs/ --watch 440 1320   # also track 440/1320 Hz
```

**From the UI:** the player has an **Export Excel** button in the transport row
(or press **E**). It opens a native Save-as dialog and reports the result under
the button. Playback pauses for the dialog and resumes afterwards, and the
playhead does not jump by the time the export took. The button uses
`--min-duration 0.05` by default — a button should hand back the usable list,
not the noisy one — so it can return fewer events than a bare `--xlsx` run.

**Why a workbook rather than CSVs.** The three tables are one artefact instead
of three loose files, and a spreadsheet stores real numeric types — a CSV is
text that Excel re-guesses on open, which is where scientific notation and
date-mangled values come from. Each sheet gets a frozen header row and filters
switched on, which matters when Detections runs to tens of thousands of rows. A
**Run info** sheet records the file, resolutions and every threshold used, so a
saved workbook says how it was produced.

`--csv DIR` writes the same tables as separate UTF-8+BOM CSV files instead —
use it if something downstream wants plain text. If `openpyxl` is missing, both
the CLI and the button fall back to CSV with a note rather than failing.

Three sheets, because "when a frequency is hit" means different things:

| sheet | one row per | use it for |
|---|---|---|
| `Events` | signal, start to end | **the readable summary** — usually what you want |
| `Detections` | peak per time slice | plotting, pivot tables; can be tens of thousands of rows |
| `Watch` | time slice per `--watch` frequency | "was my tone present, how strong?" — logged whether or not it was detected |

`Events` columns: `event_id, start_s, end_s, duration_s, center_hz,
peak_freq_hz, freq_low_hz, freq_high_hz, bandwidth_hz, peak_db, max_snr_db,
n_detections`. `Watch` has a `present` column (1/0) that filters directly
in a spreadsheet.

**Accuracy.** A bin is a hit when it sits `--snr` dB (default 12) above the
noise floor *of its own time slice*, and stands `--prominence` dB (default 6)
above the valley beside it. Against a synthetic file with known content (1 kHz
1.0-3.0 s, 2.5 kHz 2.0-5.0 s, a 200 ms 3.9 kHz blip) the defaults recover all
three to within **0.05 Hz and ~50 ms**. Onsets read ~40-50 ms early because a
1024-sample FFT window straddles them: raise `--nfft` for finer frequency,
lower it for finer timing.

Noise also throws single-slice false positives that scrape the threshold — 9 of
them on that test file. `--min-duration 0.05` drops all 9 and leaves exactly the
three real signals; it is the first knob to reach for:

```bash
./wav-player recording.wav --xlsx logs/ --min-duration 0.05 --no-detections
```

**Speed.** A 3-minute 16 kHz file analyses and saves in ~1.8 s (0.2 s
spectrogram, 0.7 s detection, 0.9 s workbook). From the player's button the
spectrogram already exists, so it is ~1.9 s. Detection was originally 4.8 s of
that: prominence was measured by walking the band in a Python loop, which for
an isolated strong tone runs the full width — 9.0M interpreter-level `min()`
calls on that file. It is numpy slicing now, bit-for-bit identical output at
7.8x the speed. Cost scales with file length, so `--no-detections` is the lever
on very long recordings: it is the bulk sheet, not the useful one.

Other knobs: `--max-gap` (bridge dropouts when merging detections into one
event, default 0.15 s), `--min-separation` (discard the weaker of two close
peaks), `--watch-tol` (band read around each watched frequency),
`--csv-max-rows` (average down the time axis on long files).

Peaks are picked by prominence rather than by thresholding contiguous spans.
A loud carrier holds ~50 bins above a 12 dB threshold through spectral leakage
alone, so a span-based rule merges it with every signal it reaches and reports
one 2 kHz-wide blob per slice; prominence keeps tones separate however wide
their skirts are.

### 3-D player view

`--view 3d` swaps the player's main panel for a live amplitude relief — playing,
seekable and scrubbable. By default the axes are **time across (X), amplitude
into the page (Y, low on the left), frequency as height (Z)**, and every axis is
reconfigurable at runtime (below).

```bash
./wav-player recording.wav --view 3d                 # macOS / Linux
.\wav-player recording.wav --view 3d                 # Windows
python -m wav_spectrogram recording.wav --view 3d    # any platform
```

**Reconfiguring the axes.** Buttons sit down each side of the plot — **flip X/Y/Z**
on the left, **swap X/Y**, **swap Y/Z**, **swap X/Z** and **reset** on the right —
so any of the three quantities (time, amplitude, frequency) can be moved onto any
spatial axis, and any axis flipped, while it plays. The keys `7`/`8`/`9` swap the
same pairs, `x`/`y`/`z` flip an axis, and `0` resets to the default. The current
mapping is shown on the status line (e.g. `X:time  Y:-amp  Z:freq`, where a
leading `-` means inverted).

**Exporting stills.** The **Export PNGs** button (transport row) writes two
images beside a name you choose: `<name>_spectrogram.png` (the flat 2-D
spectrogram) and `<name>_surface.png` (the 3-D view). The 3-D still mirrors
whatever axis layout — swaps, flips, camera angle — is on screen at the time.

The window is drawn as a stack of filled spectrum traces, one per time slice,
rather than a surface. This is a performance decision, and it drove the whole
design — measured per frame:

| approach | 48-60 slices | export resolution |
|---|---|---|
| `plot_surface` | ~250 ms (4 fps) | ~5.9 s |
| ribbons, full redraw | ~250 ms (4 fps) | — |
| **ribbons, blitted** | **~19 ms (52 fps)** | — |

Two consequences worth knowing:

* **The time axis is relative to the window** (0 at the oldest visible edge),
  not absolute file time. Blitting is what buys the 13x, and it requires the
  cached background — panes, grid, tick labels — to stay valid, so the axis
  limits must not move. Absolute position is in the status readout and scrub
  bar instead.
* **Clicking the main view does not seek** in 3-D: X is window-relative time,
  and a drag there rotates the camera. Seek with the scrub bar or the transport
  keys, which work exactly as in 2-D.

Colour runs along time, dark (oldest) to bright (newest). Amplitude already has
its own spatial axis, and re-encoding it as colour flattens the whole plot to one
shade on steady material; a recency ramp instead separates overlapping traces.
Rotate freely with the mouse — the background is recaptured after any camera
move. `--traces` (default 48) is the main cost knob; `--trace-bins`
(default 192) sets frequency resolution per trace. `--view 3d` sets the *initial*
layout only; use the side buttons or `7/8/9 · x/y/z · 0` to change it live.

The 2-D view remains the default: it is faster, and clicking it seeks.

### Player controls

(In 2-D, the default.) The main view is a live waterfall: frequency across, time downward. The newest
audio is at the bottom edge and older audio slides up and off the top, so the
image builds the way a live spectrogram prints. The vertical strip on the right
is the whole file *and* the scrub bar, sharing the same downward time axis —
click or drag it to seek; the shaded band is what the main view is showing.

| | |
|---|---|
| `space` | play / pause |
| `J` `K` `L` | reverse / pause / forward (as in video editors) |
| `R` | flip direction |
| `←` `→` | step 1 s (`shift` for 5 s) |
| `↑` `↓` | speed: 0.1x … 8x |
| `,` `.` | nudge contrast floor |
| `home` `end` | jump to start / end |
| click main view | seek there (2-D only) |
| `7` `8` `9` | 3-D: swap axes X/Y, Y/Z, X/Z |
| `x` `y` `z` | 3-D: flip that axis |
| `0` | 3-D: reset axes to default |

Buttons across the bottom mirror these: `|<  <<  <  ||  >  >>  >|` plus `-`/`+`
for speed, `Export Excel`, `Export PNGs`, and — in app mode — `Open…` and
`Close`. In `--view 3d` the axis flip/swap/reset buttons run down each side of
the plot. Closing the window also exits.

Flags still apply in app mode and carry across every file you open in that
session, so `./wav-player --nfft 4096 --seconds 12` sets up the analysis once and
then you just keep picking files.

**Why the player pauses before opening:** seeking and reverse need random access,
so the whole spectrogram is computed up front (progress is printed) and playback
is then just a moving window into that matrix — scrubbing is instant. Contrast is
also computed once over the whole file, so the image never pulses as loud content
goes past. `--player-max-rows` bounds the matrix (default 40000 rows, ~80 MB at
513 bins); past that, frames are averaged rather than dropped, so a long file
gets a smoother, slightly coarser view instead of a gappy one. Use `--stream` if
you would rather not wait at all.

### Tuning what you see

* `--nfft` — trades frequency resolution against time resolution. 1024 (default)
  is a good balance; 4096+ resolves close tones, 256 sharpens transients.
* `--overlap` — how much consecutive frames share (default 0.75). Higher scrolls
  more smoothly and costs more CPU.
* `--avg N` — power-averages N frames per row. Smooths a speckly noise floor at
  the cost of time resolution; the labeling tool's `n_avg_frames`.
* `--clim-mode` — contrast rule. `floor` (default) anchors the bottom of the
  colour range just under the noise-floor median so noise renders dark and
  signals pop. `percentile` reproduces the labeling tool's 5/95 rule; that suits
  its signal-dense captures, but on a typical WAV (mostly noise) 5/95 spans only
  the noise and the floor fills the colormap as speckle.
* `--vmin/--vmax` — pin either limit in dB to disable adaptation. Do this when
  images must be comparable across a dataset.

## Consistency with the rest of the project

The dB scaling matches `spectrogram-labeling-tool/spectrogram/utils.py` and
`vita49_pipeline`: `np.fft.fft(..., norm="forward")` then
`10*log10(10*|X|**2)`. A `--vmin/--vmax` pair chosen for one of those tools means
the same thing here, and `--export --yolo` writes the same
decoration-free square image `capture_spectrogram.py --zoom` does.

## Layout

| file | role |
|---|---|
| `wav_source.py` | streaming WAV reader; PCM decode, mono mixdown, I+jQ assembly |
| `stft.py` | `StftEngine` — overlapping STFT over an unbounded stream; `analyze` (player precompute) and `spectrogram_of` (export) for whole files |
| `player.py` | transport UI: main view, scrub strip, buttons, keyboard |
| `app.py` | launcher screen, native file dialog, app loop |
| `waterfall.py` | forward-only scrolling display (`--stream`) |
| `surface3d.py` | `RibbonView` — the player's blitted 3-D main view (`--view 3d`) |
| `export.py` | still PNG renderers (annotated preview, 3-D surface, bare YOLO tile) |
| `__main__.py` | CLI |

The pieces are independent: `StftEngine` takes any stream of blocks, so pointing
`--stream` at live UDP from `vita49_pipeline` instead of a file is just swapping
the source that feeds `push()`. `SpectrogramPlayer` takes a plain
`(time, freq)` dB matrix, so it will play anything you can put in an array —
including a cached matrix from the labeling tool.

## Notes

* The player and `--stream` need an interactive matplotlib backend (on macOS the
  default `macosx` is fine). Over SSH or in a headless shell, use `--export`.
* App mode uses **no Tk**. The launcher is a matplotlib figure — one GUI stack
  in the process, the same one the player and the labeling tool already use —
  and the file dialog is the native Finder one via `osascript`. tkinter was
  tried first and abandoned: macOS ships a deprecated Tk 8.5 that renders ttk
  windows blank, and matplotlib's Tk backend creates its own Tk root per figure,
  which invalidates widgets belonging to any other root
  (`invalid command name ".!progressbar"`). On non-macOS, `pick_file` falls back
  to tkinter, which is fine where Tk is current.
* The last folder and your six most recent files are remembered in
  `~/.wav_spectrogram.json`.
* The player window has no matplotlib nav toolbar on purpose: its pan/zoom modes
  swallow click-to-seek, and the view rewrites its own x-limits every frame.
* Playback is paced by wall clock, not by frame count, so `--speed 4` is
  genuinely 4x and the rate is the same on a fast or slow machine. A machine
  that cannot keep up drops frames, not sync.
* `--stream` costs constant memory at any file length — reading, STFT and
  display are all streaming. The player trades that for random access, bounded
  by `--player-max-rows`. `--export` caps rows with `--max-rows` (default 2048).
* Both the player and `--stream` run time vertically. They differ in direction:
  the player puts the newest audio at the *bottom* and slides older audio up;
  `--stream` pins the newest row at the top and scrolls down. The player's
  y axis is inverted rather than its data flipped, so every time value in the
  code stays a real file timestamp.
