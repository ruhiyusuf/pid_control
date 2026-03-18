#!/usr/bin/env python3
"""
soliton_step_report.py

Edit the USER SETTINGS section only.
Then run:

    python soliton_step_report.py

No command-line arguments needed.

What it does:
- loads one sweep_run_*.json file
- loads all scope_data/*.npy acquisitions for one chosen channel
- reconstructs global time using: local_time + (timestamp - t0)
- searches each expected sweep-step window for quiet plateaus
- reports the longest detected plateau overall + per step
- saves:
    * full_trace_overview.png
    * longest_plateau_zoom.png
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# ============================================================
# USER SETTINGS — EDIT THESE ONLY
# ============================================================

# Folder containing your .npy files
SCOPE_DATA_DIR = r"C:\Users\ruhiy\projects\single_soliton\scope_data"

# Sweep JSON file
SWEEP_JSON_FILE = r"C:\Users\ruhiy\projects\single_soliton\rp\sweep_run_20260317_124846.json"

# Which scope channel to analyze
CHANNEL_TO_ANALYZE = 1

# Output folder for plots + report
OUTPUT_DIR = r"C:\Users\ruhiy\projects\single_soliton\soliton_step_report_output"

# Detection tuning
SMOOTH_WINDOW_US = 0.5          # smoothing window in microseconds
ROLLING_STD_WINDOW_US = 1.0     # rolling std window in microseconds
QUIET_STD_SCALE = 0.20          # threshold = QUIET_STD_SCALE * std(signal)
MIN_PLATEAU_US = 0.50           # minimum plateau duration in microseconds
MERGE_GAP_US = 0.20             # merge nearby quiet regions if separated by small gaps
EDGE_MARGIN_US = 0.20           # ignore tiny edge regions at window boundaries


# ============================================================
# HELPERS
# ============================================================

def load_sweep_json(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


def load_scope_channel(scope_dir, channel):
    scope_dir = Path(scope_dir)
    files = sorted(scope_dir.glob(f"ch{channel}_acq*.npy"))

    if not files:
        raise FileNotFoundError(
            f"No files found for channel {channel} in {scope_dir}"
        )

    t_global_all = []
    v_global_all = []
    t0 = None

    for f in files:
        d = np.load(f, allow_pickle=True).item()

        if int(d["channel"]) != channel:
            continue

        timestamp = float(d["timestamp"])
        if t0 is None:
            t0 = timestamp

        t_local = np.asarray(d["time"], dtype=float)
        v = np.asarray(d["voltage"], dtype=float)

        t_global = t_local + (timestamp - t0)

        t_global_all.append(t_global)
        v_global_all.append(v)

    if not t_global_all:
        raise RuntimeError(f"Found files, but no usable data for channel {channel}")

    t = np.concatenate(t_global_all)
    v = np.concatenate(v_global_all)

    # sort in case timestamps/files were slightly out of order
    order = np.argsort(t)
    return t[order], v[order]


def contiguous_true_regions(mask):
    """Return list of (start_idx, end_idx) inclusive-exclusive for True runs."""
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []

    padded = np.r_[False, mask, False].astype(int)
    diff = np.diff(padded)

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))


def merge_regions_by_gap(regions, t, max_gap_s):
    if not regions:
        return []

    merged = [list(regions[0])]
    for s, e in regions[1:]:
        prev_s, prev_e = merged[-1]
        gap = t[s] - t[prev_e - 1]
        if gap <= max_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    return [tuple(x) for x in merged]


def detect_quiet_plateaus(t, v):
    """
    Detect quiet plateau regions from a 1D trace.
    Returns list of dicts with plateau info.
    """
    if len(t) < 10:
        return [], v

    dt = np.median(np.diff(t))
    sample_rate = 1.0 / dt

    smooth_window = max(11, int((SMOOTH_WINDOW_US * 1e-6) * sample_rate))
    if smooth_window % 2 == 0:
        smooth_window += 1
    if smooth_window >= len(v):
        smooth_window = len(v) - 1 if len(v) % 2 == 0 else len(v)
    if smooth_window < 5:
        smooth_window = 5

    smooth_v = savgol_filter(v, smooth_window, 3 if smooth_window >= 7 else 2)

    std_window = max(5, int((ROLLING_STD_WINDOW_US * 1e-6) * sample_rate))
    rolling_std = np.zeros_like(smooth_v)
    for i in range(len(smooth_v)):
        a = max(0, i - std_window)
        rolling_std[i] = np.std(smooth_v[a:i + 1])

    quiet_thresh = QUIET_STD_SCALE * np.std(smooth_v)
    quiet_mask = rolling_std <= quiet_thresh

    regions = contiguous_true_regions(quiet_mask)
    regions = merge_regions_by_gap(regions, t, MERGE_GAP_US * 1e-6)

    min_duration_s = MIN_PLATEAU_US * 1e-6
    edge_margin_s = EDGE_MARGIN_US * 1e-6

    plateaus = []
    tmin = t[0]
    tmax = t[-1]

    for s, e in regions:
        if e - s < 2:
            continue

        start_t = t[s]
        end_t = t[e - 1]
        dur = end_t - start_t

        if dur < min_duration_s:
            continue

        # reject tiny regions hugging window edges
        if (start_t - tmin) < edge_margin_s or (tmax - end_t) < edge_margin_s:
            continue

        plateaus.append({
            "start_idx": s,
            "end_idx": e,
            "start_t": start_t,
            "end_t": end_t,
            "duration_s": dur,
            "mean_v": float(np.mean(v[s:e])),
            "std_v": float(np.std(v[s:e])),
        })

    plateaus.sort(key=lambda x: x["duration_s"], reverse=True)
    return plateaus, smooth_v


def slice_window(t, v, t0, t1):
    mask = (t >= t0) & (t <= t1)
    return t[mask], v[mask], mask


# ============================================================
# MAIN
# ============================================================

def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Loading sweep JSON...")
    sweep = load_sweep_json(SWEEP_JSON_FILE)
    step_log = sweep["step_log"]
    print(f"Run ID: {sweep.get('run_id', 'unknown')}")
    print(f"Steps in log: {len(step_log)}")

    print("\nLoading scope data...")
    t, v = load_scope_channel(SCOPE_DATA_DIR, CHANNEL_TO_ANALYZE)
    print(f"Loaded {len(t):,} samples from channel {CHANNEL_TO_ANALYZE}")
    print(f"Global time span: {t[0]:.6f} s to {t[-1]:.6f} s")

    per_step_results = []
    overall_best = None
    overview_plateaus = []

    print("\nAnalyzing expected step windows...")
    for step in step_log:
        step_idx = step["step_idx"]
        ws = float(step["t_start_expected_s"])
        we = float(step["t_end_expected_s"])

        tw, vw, _ = slice_window(t, v, ws, we)

        if len(tw) < 10:
            per_step_results.append({
                "step_idx": step_idx,
                "window_start_s": ws,
                "window_end_s": we,
                "best_plateau": None,
            })
            continue

        plateaus, smooth_v = detect_quiet_plateaus(tw, vw)
        best = plateaus[0] if plateaus else None

        if best is not None:
            best_global = best.copy()
            overview_plateaus.append((best["start_t"], best["end_t"], step_idx))

            if overall_best is None or best["duration_s"] > overall_best["duration_s"]:
                overall_best = {
                    **best_global,
                    "step_idx": step_idx,
                    "window_start_s": ws,
                    "window_end_s": we,
                    "t_window": tw,
                    "v_window": vw,
                    "smooth_window": smooth_v,
                }

        per_step_results.append({
            "step_idx": step_idx,
            "window_start_s": ws,
            "window_end_s": we,
            "best_plateau": best,
        })

    # --------------------------------------------------------
    # Write text report
    # --------------------------------------------------------
    report_path = outdir / "soliton_step_report.txt"
    with open(report_path, "w") as f:
        f.write("Soliton Step Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Sweep JSON: {SWEEP_JSON_FILE}\n")
        f.write(f"Scope dir:   {SCOPE_DATA_DIR}\n")
        f.write(f"Channel:     {CHANNEL_TO_ANALYZE}\n")
        f.write(f"Samples:     {len(t)}\n")
        f.write(f"Global span: {t[0]:.6f} s to {t[-1]:.6f} s\n\n")

        if overall_best is None:
            f.write("No plateau detected in any expected step window.\n")
        else:
            f.write("Longest detected plateau overall\n")
            f.write("-" * 70 + "\n")
            f.write(f"Step index:   {overall_best['step_idx']}\n")
            f.write(f"Start time:   {overall_best['start_t']:.9f} s\n")
            f.write(f"End time:     {overall_best['end_t']:.9f} s\n")
            f.write(f"Duration:     {overall_best['duration_s'] * 1e6:.3f} us\n")
            f.write(f"Mean voltage: {overall_best['mean_v']:.6f} V\n")
            f.write(f"Std voltage:  {overall_best['std_v']:.6f} V\n\n")

        f.write("Per-step summary\n")
        f.write("-" * 70 + "\n")
        for row in per_step_results:
            step_idx = row["step_idx"]
            ws = row["window_start_s"]
            we = row["window_end_s"]
            best = row["best_plateau"]

            if best is None:
                f.write(
                    f"Step {step_idx:02d}: window [{ws:.6f}, {we:.6f}] s -> no plateau detected\n"
                )
            else:
                f.write(
                    f"Step {step_idx:02d}: window [{ws:.6f}, {we:.6f}] s -> "
                    f"plateau [{best['start_t']:.9f}, {best['end_t']:.9f}] s, "
                    f"{best['duration_s'] * 1e6:.3f} us\n"
                )

    print(f"\nSaved report: {report_path}")

    # --------------------------------------------------------
    # Overview plot
    # --------------------------------------------------------
    plt.figure(figsize=(14, 5))
    plt.plot(t, v, linewidth=0.8, label=f"CH{CHANNEL_TO_ANALYZE}")

    for step in step_log:
        ws = float(step["t_start_expected_s"])
        we = float(step["t_end_expected_s"])
        plt.axvspan(ws, we, alpha=0.08)

    for ps, pe, step_idx in overview_plateaus:
        plt.axvspan(ps, pe, alpha=0.35)
        plt.text(
            0.5 * (ps + pe),
            np.nanmax(v),
            f"{step_idx}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90
        )

    if overall_best is not None:
        plt.axvline(overall_best["start_t"], linestyle="--", linewidth=1.5)
        plt.axvline(overall_best["end_t"], linestyle="--", linewidth=1.5)

    plt.xlabel("Global Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Full Trace Overview with Expected Step Windows and Detected Plateaus")
    plt.grid(True)

    overview_png = outdir / "full_trace_overview.png"
    plt.tight_layout()
    plt.savefig(overview_png, dpi=200)
    plt.close()
    print(f"Saved plot:   {overview_png}")

    # --------------------------------------------------------
    # Zoom plot for longest plateau
    # --------------------------------------------------------
    if overall_best is not None:
        ps = overall_best["start_t"]
        pe = overall_best["end_t"]
        pad = max(20e-6, 2 * (pe - ps))

        mask = (t >= ps - pad) & (t <= pe + pad)
        tz = t[mask]
        vz = v[mask]

        _, smooth_z = detect_quiet_plateaus(tz, vz)

        plt.figure(figsize=(12, 5))
        plt.plot(tz, vz, linewidth=0.9, label="Raw")
        plt.plot(tz, smooth_z, linewidth=1.5, label="Smoothed")
        plt.axvspan(ps, pe, alpha=0.30, label="Longest plateau")
        plt.axvline(ps, linestyle="--", linewidth=1.2)
        plt.axvline(pe, linestyle="--", linewidth=1.2)

        plt.xlabel("Global Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title(
            f"Zoom Around Longest Plateau (Step {overall_best['step_idx']}, "
            f"{overall_best['duration_s'] * 1e6:.3f} us)"
        )
        plt.grid(True)
        plt.legend()

        zoom_png = outdir / "longest_plateau_zoom.png"
        plt.tight_layout()
        plt.savefig(zoom_png, dpi=200)
        plt.close()
        print(f"Saved plot:   {zoom_png}")

        print("\nLongest detected plateau overall:")
        print(f"  Step:      {overall_best['step_idx']}")
        print(f"  Start:     {overall_best['start_t']:.9f} s")
        print(f"  End:       {overall_best['end_t']:.9f} s")
        print(f"  Duration:  {overall_best['duration_s'] * 1e6:.3f} us")
        print(f"  Mean V:    {overall_best['mean_v']:.6f} V")
        print(f"  Std V:     {overall_best['std_v']:.6f} V")
    else:
        print("\nNo quiet plateau detected in the expected step windows.")

    print("\nDone.")


if __name__ == "__main__":
    main()
