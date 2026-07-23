"""CLI for wav_spectrogram.

    python -m wav_spectrogram                              # app: pick a file
    python -m wav_spectrogram song.wav                      # player
    python -m wav_spectrogram cap.wav --mode iq --center 433.92e6
    python -m wav_spectrogram song.wav --export out/        # PNGs instead
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .stft import StftEngine, analyze, spectrogram_of
from .wav_source import WavSource


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wav_spectrogram",
        description="Spectrogram player for WAV files — play, seek, reverse "
                    "and scrub. Also a streaming waterfall (--stream) and "
                    "headless PNG export (--export).")
    p.add_argument("wav", type=Path, nargs="?",
                   help="input .wav file; omit it to launch the app "
                        "with a file picker")

    p.add_argument("--mode", choices=["auto", "audio", "iq"], default="auto",
                   help="interpret the file as real audio or 2-channel I/Q "
                        "(default: auto — iq only for 2-ch files with --center "
                        "or an iq/baseband filename)")
    p.add_argument("--center", type=float, default=0.0, metavar="HZ",
                   help="RF centre frequency for IQ files, e.g. 433.92e6")

    g = p.add_argument_group("analysis")
    g.add_argument("--nfft", type=int, default=1024,
                   help="FFT size — bigger = finer frequency, coarser time (default 1024)")
    g.add_argument("--overlap", type=float, default=0.75, metavar="FRAC",
                   help="frame overlap 0..0.95 — higher = smoother scroll (default 0.75)")
    g.add_argument("--avg", type=int, default=1,
                   help="power-average N frames per row; smooths the noise floor (default 1)")

    g = p.add_argument_group("display")
    g.add_argument("--stream", action="store_true",
                   help="forward-only scrolling waterfall instead of the player "
                        "(no seek/reverse, but starts instantly on huge files)")
    g.add_argument("--view", choices=["2d", "3d"], default="2d",
                   help="player main view: 2d spectrogram image (default) or "
                        "3d amplitude relief (time across, frequency into the "
                        "page, amplitude as height)")
    g.add_argument("--traces", type=int, default=48, metavar="N",
                   help="--view 3d: time slices drawn per frame; the main "
                        "cost knob, lower = faster (default 48)")
    g.add_argument("--trace-bins", type=int, default=192, metavar="N",
                   help="--view 3d: frequency points per trace (default 192)")
    g.add_argument("--loop", action="store_true",
                   help="player: restart at the far end instead of stopping")
    g.add_argument("--seconds", type=float, default=6.0,
                   help="seconds of spectrogram visible at once (default 6)")
    g.add_argument("--speed", type=float, default=1.0,
                   help="playback rate: 2 = twice real time (default 1)")
    g.add_argument("--fps", type=int, default=30, help="redraw rate (default 30)")
    g.add_argument("--cmap", default="magma",
                   help="colormap (default magma, matching the labeling tool)")
    g.add_argument("--vmin", type=float, default=None,
                   help="pin the lower dB limit (default: adaptive, see --clim-mode)")
    g.add_argument("--vmax", type=float, default=None,
                   help="pin the upper dB limit (default: adaptive, see --clim-mode)")
    g.add_argument("--clim-mode", choices=["floor", "percentile"], default="floor",
                   help="contrast rule: floor = noise-floor referenced (default), "
                        "percentile = the labeling tool's 5/95 rule")
    g.add_argument("--save-last", type=Path, default=None, metavar="PNG",
                   help="write the final waterfall frame here when playback ends")

    g = p.add_argument_group("spreadsheet analysis (no live window)")
    g.add_argument("--xlsx", type=Path, default=None, metavar="DIR",
                   help="scan the whole file and write one .xlsx workbook to "
                        "DIR (Events / Detections / Watch sheets), then exit")
    g.add_argument("--csv", type=Path, default=None, metavar="DIR",
                   help="same analysis as separate .csv files in DIR — use "
                        "when you want plain text rather than a workbook")
    g.add_argument("--snr", type=float, default=12.0, metavar="DB",
                   help="a bin counts as a hit this far above the noise floor "
                        "of its own time slice (default 12)")
    g.add_argument("--prominence", type=float, default=6.0, metavar="DB",
                   help="a peak must stand this far above the valley beside it "
                        "to count, which rejects shoulders on a loud signal's "
                        "leakage skirt (default 6)")
    g.add_argument("--min-separation", type=float, default=0.0, metavar="HZ",
                   help="discard the weaker of two peaks closer than this "
                        "(default 0 = no limit)")
    g.add_argument("--watch", type=float, nargs="+", default=None, metavar="HZ",
                   help="also log these specific frequencies over time, hit or "
                        "not, e.g. --watch 440 1320")
    g.add_argument("--watch-tol", type=float, default=None, metavar="HZ",
                   help="half-width of the band read around each --watch "
                        "frequency (default: one FFT bin)")
    g.add_argument("--min-duration", type=float, default=0.0, metavar="S",
                   help="drop events shorter than this from events.csv "
                        "(default 0 = keep all)")
    g.add_argument("--max-gap", type=float, default=0.15, metavar="S",
                   help="bridge dropouts up to this long when merging "
                        "detections into one event (default 0.15)")
    g.add_argument("--no-detections", action="store_true",
                   help="skip the big per-slice detections.csv; write only "
                        "events (and any --watch log)")
    g.add_argument("--csv-max-rows", type=int, default=0, metavar="N",
                   help="time resolution cap for --csv; 0 (default) = full "
                        "resolution, no averaging")

    g = p.add_argument_group("export (no live window)")
    g.add_argument("--export", type=Path, default=None, metavar="DIR",
                   help="render the whole file to PNGs in DIR and exit")
    g.add_argument("--surface", action="store_true",
                   help="with --export: also write a 3-D surface (time across, "
                        "frequency into the page, amplitude as height)")
    g.add_argument("--surface-view", nargs=2, type=float, default=(35.0, -60.0),
                   metavar=("ELEV", "AZIM"),
                   help="camera angle for --surface (default 35 -60)")
    g.add_argument("--yolo", action="store_true",
                   help="with --export: also write a bare square YOLO image")
    g.add_argument("--imgsz", type=int, default=640, help="YOLO image size (default 640)")
    g.add_argument("--gray", action="store_true", help="grayscale instead of a colormap")
    g.add_argument("--max-rows", type=int, default=2048,
                   help="cap time rows in an export (default 2048)")
    g.add_argument("--player-max-rows", type=int, default=40000, metavar="N",
                   help="memory bound for the player's precomputed matrix; "
                        "beyond this, frames are averaged (default 40000)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not 0.0 <= args.overlap < 0.96:
        print("--overlap must be in [0, 0.95]", file=sys.stderr)
        return 2
    hop = max(int(args.nfft * (1.0 - args.overlap)), 1)

    if args.wav is None:                          # no filename -> app mode
        from .app import run_app
        return run_app(args, hop)

    try:
        source = WavSource(args.wav, mode=args.mode, center_hz=args.center)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(source.describe())

    if args.csv or args.xlsx:
        from .detect import build_sheets, write_csv, write_xlsx, xlsx_available
        out_dir = args.xlsx or args.csv
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            with source:
                psd, freqs, times = spectrogram_of(
                    source, n_fft=args.nfft, hop=hop, avg=args.avg,
                    max_rows=args.csv_max_rows or None)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        stem = args.wav.stem
        sheets = build_sheets(
            psd, freqs, times, snr_db=args.snr, prominence_db=args.prominence,
            min_separation_hz=args.min_separation, max_gap_s=args.max_gap,
            min_duration_s=args.min_duration, watch_hz=args.watch,
            watch_tol=args.watch_tol, include_detections=not args.no_detections)

        # --csv is the explicit opt-out; otherwise write a workbook, falling
        # back to CSV when openpyxl is absent rather than failing the run.
        want_xlsx = bool(args.xlsx) or (not args.csv)
        if want_xlsx and not xlsx_available():
            print("note: openpyxl not installed - writing CSV instead "
                  "(pip install openpyxl for .xlsx)", file=sys.stderr)
            want_xlsx = False

        if want_xlsx:
            info = {
                "source file": str(args.wav),
                "sample rate (Hz)": source.sample_rate,
                "duration (s)": round(source.duration, 3),
                "fft size": args.nfft,
                "overlap": args.overlap,
                "time resolution (ms)": round(
                    float(times[1] - times[0]) * 1000, 2) if len(times) > 1 else 0.0,
                "frequency resolution (Hz)": round(float(freqs[1] - freqs[0]), 3),
                "snr threshold (dB)": args.snr,
                "prominence threshold (dB)": args.prominence,
                "min event duration (s)": args.min_duration,
                "max gap (s)": args.max_gap,
            }
            path = out_dir / f"{stem}_analysis.xlsx"
            counts = write_xlsx(path, sheets, info=info)
            summary = ", ".join(f"{n} {t.lower()}" for t, n in counts.items())
            print(f"wrote {path}  ({summary})")
        else:
            for title, rows, columns in sheets:
                path = out_dir / f"{stem}_{title.lower()}.csv"
                n = write_csv(rows, path, columns=columns)
                print(f"wrote {path}  ({n} {title.lower()})")

        res = times[1] - times[0] if len(times) > 1 else 0.0
        print(f"scanned {psd.shape[0]} slices x {psd.shape[1]} bins "
              f"({res*1000:.1f} ms/slice, {freqs[1]-freqs[0]:.1f} Hz/bin)")
        return 0

    if args.export:
        from .export import render_preview, render_surface, render_yolo
        args.export.mkdir(parents=True, exist_ok=True)
        try:
            with source:
                psd, freqs, times = spectrogram_of(
                    source, n_fft=args.nfft, hop=hop, avg=args.avg,
                    max_rows=args.max_rows)
        except RuntimeError as exc:               # file shorter than one row
            print(f"error: {exc}", file=sys.stderr)
            return 1
        stem = args.wav.stem
        preview = args.export / f"{stem}_spectrogram.png"
        render_preview(psd, freqs, times, preview,
                       title=f"{args.wav.name}  ({source.duration:.1f}s, "
                             f"{source.sample_rate/1e3:.1f} kHz)",
                       cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
                       is_iq=source.is_iq, clim_mode=args.clim_mode)
        print(f"wrote {preview}  ({psd.shape[0]} rows x {psd.shape[1]} bins)")
        if args.surface:
            surf = args.export / f"{stem}_surface.png"
            render_surface(psd, freqs, times, surf,
                           title=f"{args.wav.name}  ({source.duration:.1f}s, "
                                 f"{source.sample_rate/1e3:.1f} kHz)",
                           cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
                           is_iq=source.is_iq, clim_mode=args.clim_mode,
                           elev=args.surface_view[0], azim=args.surface_view[1])
            print(f"wrote {surf}  (3-D surface)")
        if args.yolo:
            tile = args.export / f"{stem}_yolo.png"
            render_yolo(psd, tile, imgsz=args.imgsz, gray=args.gray,
                        cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
                        clim_mode=args.clim_mode)
            print(f"wrote {tile}  ({args.imgsz}x{args.imgsz})")
        return 0

    if args.stream:
        engine = StftEngine(source.sample_rate, n_fft=args.nfft, hop=hop,
                            is_iq=source.is_iq, center_hz=args.center, avg=args.avg)
        print(f"stream: {engine.n_bins} bins, {1/engine.seconds_per_row:.1f} rows/s, "
              f"{args.seconds:.0f}s window — close the window to stop")
        from .waterfall import run_live
        with source:
            run_live(source, engine, seconds=args.seconds, speed=args.speed,
                     cmap=args.cmap, vmin=args.vmin, vmax=args.vmax, fps=args.fps,
                     save_last=args.save_last, clim_mode=args.clim_mode)
        return 0

    # Player: precompute, then play. Progress matters here because a long file
    # takes a moment and an unexplained pause looks like a hang.
    def show_progress(frac):
        bar = int(frac * 30)
        print(f"\ranalysing [{'#' * bar}{'.' * (30 - bar)}] {frac*100:3.0f}%",
              end="", flush=True)

    try:
        with source:
            psd, freqs, row_s = analyze(
                source, n_fft=args.nfft, hop=hop, avg=args.avg,
                max_rows=args.player_max_rows, progress=show_progress)
    except RuntimeError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    print(f"\r{psd.shape[0]} rows x {psd.shape[1]} bins "
          f"({psd.nbytes/1e6:.0f} MB, {1/row_s:.0f} rows/s)" + " " * 20)

    from .player import SpectrogramPlayer
    SpectrogramPlayer(
        psd, freqs, row_s, window_s=args.seconds, cmap=args.cmap,
        vmin=args.vmin, vmax=args.vmax, clim_mode=args.clim_mode,
        is_iq=source.is_iq, loop=args.loop, speed=args.speed,
        view=args.view, traces=args.traces, trace_bins=args.trace_bins,
        csv_stem=args.wav.stem, csv_snr=args.snr,
        csv_min_duration=args.min_duration or 0.05,
        title=f"{args.wav.name} — {source.sample_rate/1e3:.1f} kHz"
              f"{' IQ' if source.is_iq else ''}",
    ).show(fps=args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
