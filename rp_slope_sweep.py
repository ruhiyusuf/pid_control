import numpy as np
import dearpygui.dearpygui as dpg
import socket, struct
import json, os

# ============================================================
# Red Pitaya protocol
# Packet format:
#   [uint32 nbytes][uint8 channel][float32 duration_s][int16 payload]
# Sent in BIG-endian for header fields.
# Payload is int16 native-endian (matches your server's np.int16).
# ============================================================

SAVE_PATH = "waveform_settings.json"

TIME_UNITS = {"us": 1e-6, "ms": 1e-3, "s": 1.0}
TIME_UNIT_ORDER = ["us", "ms", "s"]

# ---------------------------
# Parameters (internal storage is SI seconds for *_s)
# ---------------------------
params = {
    "rp_ip": "rp-f0cbdd.local",
    "rp_port": 9000,
    "output_channel": 1,

    # ---- SLOPE REFERENCE (constant) ----
    # This is NOT RP playback frequency. It's only used to compute the ramp SLOPE.
    "slope_ref_hz": 5000.0,   # 5 kHz triangle slope reference
    "slope_scale": 1.0,       # optional multiplier

    # waveform levels (VOLTS in your design space)
    "v_low": -0.2,
    "v_high": 1.0,
    "v_mid": 0.0,

    # timing (seconds)
    "t_hold_high": 2e-3,
    "t_hold_mid": 2e-3,

    # UI time unit (display only)
    "time_unit": "ms",
}

# initial and final settings for sweep
waveform1_params = params.copy()
waveform2_params = params.copy()

# sweep mode toggles
sweep_voltage_enabled = True
sweep_time_enabled = True

# global timing (used when sweeping voltage ONLY and NOT sweeping time)
global_timing = {
    "t_hold_high": float(params["t_hold_high"]),
    "t_hold_mid": float(params["t_hold_mid"]),
}

# increments
increments = {
    "v_low": 0.02,
    "v_high": 0.02,
    "v_mid": 0.02,
    "t_hold_high": 0.2e-3,
    "t_hold_mid": 0.2e-3,
}

# plotting limits
y_min, y_max = -0.5, 1.2

# sweep preview storage
segments = []
segment_times = []
saved_sweep_steps = []
segment_bounds = []
x_vals = []
full_waveform = None
combined_freq = None

last_total_time = None

# sampling
SAMPLES_PER_SEG = 2048

# ---- IMPORTANT: your observed crash threshold ----
# You said it only segfaults AFTER something > 28672 samples.
# So cap to 16348 and DO NOT SEND more than that.
RP_ARB_LIMIT = 16348         # you can change to 28000 for headroom
MAX_POINTS_RP = RP_ARB_LIMIT # keep your existing variable name

# cached slope reference (V/s) computed from initial waveform1
SLOPE_REF_V_PER_S = None

EXCLUDE_KEYS = {"rp_ip", "rp_port", "output_channel", "slope_ref_hz", "slope_scale", "time_unit"}
VOLT_KEYS = {"v_low", "v_high", "v_mid"}
TIME_KEYS = {"t_hold_high", "t_hold_mid"}


# ============================================================
# Unit conversion helpers (UI <-> internal seconds)
# ============================================================
def _unit_to_s():
    unit = params.get("time_unit", "ms")
    return TIME_UNITS.get(unit, 1e-3)

def seconds_to_gui(val_s: float) -> float:
    return float(val_s) / _unit_to_s()

def gui_to_seconds(val_gui: float) -> float:
    return float(val_gui) * _unit_to_s()


# ============================================================
# Sweep helpers
# ============================================================
def _is_number(x):
    return isinstance(x, (int, float, np.integer, np.floating))

def _step_toward(curr, target, step):
    curr = float(curr); target = float(target); step = float(step)
    if step <= 0:
        return curr
    if target > curr:
        return min(curr + step, target)
    return max(curr - step, target)

def _active_sweep_keys():
    keys = []
    if sweep_voltage_enabled:
        keys += sorted(list(VOLT_KEYS))
    if sweep_time_enabled:
        keys += sorted(list(TIME_KEYS))
    return keys


# ============================================================
# NEW: Predict segments/points + recommend increments
# ============================================================
def estimate_num_segments():
    """
    Estimate number of segments produced by sweep loop.
    Sweep loop terminates when ALL keys hit target.
    Because you step all keys each iteration, the loop count is governed by
    the maximum steps among the active keys.
    """
    active = _active_sweep_keys()
    if not active:
        return 1

    max_steps = 0
    for key in active:
        v1 = waveform1_params.get(key, None)
        v2 = waveform2_params.get(key, None)
        inc = float(increments.get(key, 0.0))

        if v1 is None or v2 is None or (not _is_number(v1)) or (not _is_number(v2)):
            continue

        delta = abs(float(v2) - float(v1))
        if delta < 1e-12:
            steps = 0
        elif inc <= 0:
            steps = 10**12  # invalid increment
        else:
            steps = int(np.ceil(delta / inc))

        max_steps = max(max_steps, steps)

    # +1 for the initial segment
    return 1 + max_steps

def estimate_points_and_recommendations():
    """
    Returns:
      predicted_pts, nseg, max_steps_allowed, min_inc_recommendations(dict)
    """
    nseg = estimate_num_segments()
    predicted_pts = int(nseg * SAMPLES_PER_SEG)

    max_segments_allowed = max(1, RP_ARB_LIMIT // SAMPLES_PER_SEG)
    max_steps_allowed = max(0, max_segments_allowed - 1)

    rec = {}
    active = _active_sweep_keys()

    # minimum increment needed per key to not exceed max_steps_allowed
    # inc_min = delta / max_steps_allowed
    if max_steps_allowed == 0:
        # only initial segment allowed (super strict)
        for key in active:
            rec[key] = np.inf
    else:
        for key in active:
            v1 = waveform1_params.get(key, None)
            v2 = waveform2_params.get(key, None)
            if v1 is None or v2 is None or (not _is_number(v1)) or (not _is_number(v2)):
                continue
            delta = abs(float(v2) - float(v1))
            rec[key] = (delta / float(max_steps_allowed)) if delta > 0 else 0.0

    return predicted_pts, nseg, max_steps_allowed, rec

def _fmt_inc(key, val):
    if not np.isfinite(val):
        return "∞"
    if key in VOLT_KEYS:
        return f"{val:.6g} V"
    # time key: show in current GUI unit too
    return f"{val:.6g} s ({seconds_to_gui(val):.6g} {params['time_unit']})"

def update_points_estimate_ui():
    if not dpg.does_item_exist("points_estimate_text"):
        return

    predicted_pts, nseg, max_steps_allowed, rec = estimate_points_and_recommendations()
    ok = predicted_pts <= RP_ARB_LIMIT

    line1 = f"Estimated samples: {predicted_pts} / {RP_ARB_LIMIT}   ({nseg} segments x {SAMPLES_PER_SEG})"
    line2 = f"Max allowed steps (approx): {max_steps_allowed}"

    # show the *tightest* constraint key (largest delta / inc)
    active = _active_sweep_keys()
    worst_key = None
    worst_steps = -1
    for key in active:
        v1 = float(waveform1_params.get(key, 0.0))
        v2 = float(waveform2_params.get(key, 0.0))
        inc = float(increments.get(key, 0.0))
        delta = abs(v2 - v1)
        if delta < 1e-12:
            continue
        if inc <= 0:
            steps = 10**12
        else:
            steps = int(np.ceil(delta / inc))
        if steps > worst_steps:
            worst_steps = steps
            worst_key = key

    if worst_key is None:
        line3 = "Governing key: (none — already at target)"
    else:
        line3 = f"Governing key: {worst_key} (~{worst_steps} steps with current increment)"

    # recommendations: only show for active keys with delta>0
    rec_lines = []
    for key in active:
        v1 = float(waveform1_params.get(key, 0.0))
        v2 = float(waveform2_params.get(key, 0.0))
        if abs(v2 - v1) < 1e-12:
            continue
        rec_lines.append(f"  - {key}: inc ≥ {_fmt_inc(key, rec.get(key, np.inf))}")

    if not rec_lines:
        line4 = "Recommended minimum increments: (none needed)"
    else:
        line4 = "Recommended minimum increments to stay under limit:\n" + "\n".join(rec_lines)

    status = "OK" if ok else "TOO MANY POINTS"
    text = f"[{status}]\n{line1}\n{line2}\n{line3}\n\n{line4}"

    dpg.set_value("points_estimate_text", text)


# ============================================================
# Slope reference + derived rise time (SECONDS internally)
# ============================================================
def triangle_slope_from_freq(v_low, v_high, freq_hz, slope_scale=1.0):
    dv = float(v_high) - float(v_low)
    return abs(2.0 * dv * float(freq_hz)) * float(slope_scale)

def compute_rise_time_s_from_slope(v_start, v_end, slope_v_per_s):
    dv = abs(float(v_end) - float(v_start))
    if slope_v_per_s <= 1e-12:
        return 0.0
    return dv / float(slope_v_per_s)

def update_slope_reference_from_initial():
    """Compute slope ref (V/s) using INITIAL waveform1 voltages + params slope_ref_hz."""
    global SLOPE_REF_V_PER_S
    f = float(params.get("slope_ref_hz", 0.0))
    s = float(params.get("slope_scale", 1.0))
    if f <= 0:
        SLOPE_REF_V_PER_S = 0.0
        return
    v_low_init = float(waveform1_params["v_low"])
    v_high_init = float(waveform1_params["v_high"])
    SLOPE_REF_V_PER_S = triangle_slope_from_freq(v_low_init, v_high_init, f, s)

def derived_t_rise_s(p):
    """Derived rise time in SECONDS, using fixed slope reference."""
    if SLOPE_REF_V_PER_S is None:
        update_slope_reference_from_initial()
    slope = float(SLOPE_REF_V_PER_S or 0.0)
    return compute_rise_time_s_from_slope(p["v_low"], p["v_high"], slope)


# ============================================================
# Waveform generation (one cycle)
# ramp up -> hold high -> instant drop -> hold mid -> instant jump to low
# ============================================================
def _merge_globals_into_params(p):
    q = p.copy()
    # voltage-only sweep (timing not swept) => timing comes from global_timing
    if sweep_voltage_enabled and (not sweep_time_enabled):
        q["t_hold_high"] = float(global_timing["t_hold_high"])
        q["t_hold_mid"]  = float(global_timing["t_hold_mid"])
    return q

def generate_cycle_waveform(p, num_samples=16384):
    p = _merge_globals_into_params(p)

    v_low  = float(p["v_low"])
    v_high = float(p["v_high"])
    v_mid  = float(p["v_mid"])

    t_rise = max(0.0, derived_t_rise_s(p))              # SECONDS
    t_hh   = max(0.0, float(p["t_hold_high"]))        # SECONDS
    t_hm   = max(0.0, float(p["t_hold_mid"]))         # SECONDS

    total_time = max(t_rise + t_hh + t_hm, 1e-9)
    n = int(max(256, num_samples))

    t = np.linspace(0, total_time, n, endpoint=False)
    y = np.empty_like(t, dtype=float)

    t1 = t_rise
    t2 = t_rise + t_hh

    # ramp up
    if t_rise <= 1e-12:
        y[:] = v_high
    else:
        ramp_mask = t < t1
        y[ramp_mask] = v_low + (v_high - v_low) * (t[ramp_mask] / t_rise)
        y[~ramp_mask] = v_high

    # hold high
    hh_mask = (t >= t1) & (t < t2)
    y[hh_mask] = v_high

    # drop + hold mid
    hm_mask = t >= t2
    y[hm_mask] = v_mid

    freq = 1.0 / total_time
    return t, y, total_time, freq

def _seg_duration_from_t(t):
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        return float(t[-1]) if len(t) else 0.0
    dt = float(t[1] - t[0])
    return float(t[-1] + dt)  # matches your known-good sender logic


# ============================================================
# Sweep generation
# ============================================================
def generate_sweep():
    global full_waveform, combined_freq, x_vals

    update_slope_reference_from_initial()

    x_vals.clear()
    segments.clear()
    saved_sweep_steps.clear()
    segment_times.clear()
    segment_bounds.clear()

    if dpg.does_item_exist("highlight_box"):
        dpg.delete_item("highlight_box")

    dpg.set_value("sweep_plot_series", [[], []])

    active_keys = _active_sweep_keys()
    interp_params = waveform1_params.copy()

    current_time = 0.0

    # first segment: initial
    t0, y0, _, _ = generate_cycle_waveform(waveform1_params, num_samples=SAMPLES_PER_SEG)
    d0 = _seg_duration_from_t(t0)

    segments.append(y0)
    segment_times.append((0.0, d0))
    saved_sweep_steps.append(_merge_globals_into_params(waveform1_params))
    segment_bounds.append((0.0, d0, float(np.min(y0)), float(np.max(y0))))
    x_vals.extend((t0 + 0.0).tolist())
    current_time = d0

    # sweep loop
    max_steps_guard = 200000
    step_count = 0

    while True:
        step_count += 1
        if step_count > max_steps_guard:
            raise RuntimeError("Sweep exceeded max steps guard; increments too small?")

        done = True

        for key in active_keys:
            v1 = waveform1_params.get(key, None)
            v2 = waveform2_params.get(key, None)
            if v1 is None or v2 is None:
                continue
            if (not _is_number(v1)) or (not _is_number(v2)):
                continue

            inc = float(increments.get(key, 0.0))
            curr = float(interp_params.get(key, v1))
            target = float(v2)

            new_val = _step_toward(curr, target, inc)
            interp_params[key] = new_val

            if abs(new_val - target) > 1e-12:
                done = False

        t_seg, y_seg, _, _ = generate_cycle_waveform(interp_params, num_samples=SAMPLES_PER_SEG)
        d_seg = _seg_duration_from_t(t_seg)

        t_seg_shift = t_seg + current_time

        segments.append(y_seg)
        segment_times.append((current_time, current_time + d_seg))
        saved_sweep_steps.append(_merge_globals_into_params(interp_params))
        segment_bounds.append((current_time, current_time + d_seg, float(np.min(y_seg)), float(np.max(y_seg))))

        x_vals.extend(t_seg_shift.tolist())
        current_time += d_seg

        if done:
            break

    full_waveform = np.concatenate(segments)
    total_time = float(current_time)
    combined_freq = (1.0 / total_time) if total_time > 0 else 0.0

    dpg.set_value("sweep_plot_series", [np.array(x_vals), full_waveform])
    # return np.array(x_vals), full_waveform, combined_freq
    return np.array(x_vals), full_waveform, combined_freq, total_time



# ============================================================
# RP scaling + send
# ============================================================
def volts_to_i16_minmax(v):
    """Map waveform volts to [-1,1] using min/max."""
    w = np.asarray(v, dtype=float)
    vmin = float(np.min(w))
    vmax = float(np.max(w))
    if vmax - vmin < 1e-9:
        vmax = vmin + 1e-9
    w_norm = 2.0 * (w - vmin) / (vmax - vmin) - 1.0
    w_norm = np.clip(w_norm, -1.0, 1.0)
    return (w_norm * (2**13 - 1)).astype(np.int16), vmin, vmax

def rp_send_waveform_big_endian(wave_i16, duration_s, rp_ip, rp_port, ch):
    ch = int(ch)
    if ch not in (1, 2):
        raise ValueError("channel must be 1 or 2")

    wave_i16 = np.asarray(wave_i16, dtype=np.int16)
    payload = wave_i16.tobytes()
    nbytes = len(payload)

    if nbytes % 2 != 0:
        raise ValueError(f"Payload bytes must be even, got {nbytes}")

    duration_s = float(duration_s)
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError(f"Bad duration_s: {duration_s}")

    with socket.create_connection((rp_ip, int(rp_port)), timeout=5) as s:
        s.sendall(struct.pack(">I", nbytes))
        s.sendall(struct.pack(">B", ch))
        s.sendall(struct.pack(">f", duration_s))
        s.sendall(payload)

def deploy_sweep():
    global full_waveform, x_vals, segment_times, last_total_time

    if full_waveform is None or len(full_waveform) < 2:
        dpg.set_value("status_text", "Generate sweep first.")
        return

    try:
        rp_ip = str(dpg.get_value("rp_ip")).strip()
        rp_port = int(params["rp_port"])
        ch = int(dpg.get_value("output_channel"))

        w = np.array(full_waveform, dtype=float)

        # HARD CAP client-side: NEVER exceed RP_ARB_LIMIT
        if len(w) > RP_ARB_LIMIT:
            dpg.set_value(
                "status_text",
                f"Blocked: waveform has {len(w)} pts > {RP_ARB_LIMIT}. Increase increments or reduce sweep range."
            )
            return

        # duration: match known-good style
        # if len(x_vals) >= 2:
        #     dt = float(x_vals[1] - x_vals[0])
        #     duration_s = float(x_vals[-1] + dt)
        # else:
        #     duration_s = 1e-6

        # duration = sum of each segment duration
        if not segment_times:
            raise ValueError("No segments. Click Start Sweep first.")

        duration_s = float(sum((end - start) for (start, end) in segment_times))
        duration_s = max(duration_s, 1e-9)
        last_total_time = duration_s  # optional: keep it around for display/debug
        print(f"DEBUG: nseg={len(segment_times)} duration_s={duration_s}")


        # duration_s = max(duration_s, 1e-9)

        wave_i16, vmin, vmax = volts_to_i16_minmax(w)
        rp_send_waveform_big_endian(wave_i16, duration_s, rp_ip, rp_port, ch)

        slope = float(SLOPE_REF_V_PER_S or 0.0)
        dpg.set_value(
            "status_text",
            f"Sent | CH{ch} | pts={len(wave_i16)} | T={duration_s:.9f}s | f≈{(1.0/duration_s):.6f}Hz | "
            f"slope_ref={slope:.6g} V/s | mapped [{vmin:.3f},{vmax:.3f}] -> [-1,1]"
        )
    except Exception as e:
        dpg.set_value("status_text", f"RP ERROR: {e}")


# ============================================================
# GUI sync + show/hide logic
# ============================================================
def _show_item(tag, show=True):
    if dpg.does_item_exist(tag):
        dpg.configure_item(tag, show=show)

def pull_global_params():
    global sweep_voltage_enabled, sweep_time_enabled

    if dpg.does_item_exist("chk_sweep_voltage"):
        sweep_voltage_enabled = bool(dpg.get_value("chk_sweep_voltage"))
    if dpg.does_item_exist("chk_sweep_time"):
        sweep_time_enabled = bool(dpg.get_value("chk_sweep_time"))

    if dpg.does_item_exist("slope_ref_hz"):
        try:
            params["slope_ref_hz"] = float(dpg.get_value("slope_ref_hz"))
        except:
            pass
    if dpg.does_item_exist("slope_scale"):
        try:
            params["slope_scale"] = float(dpg.get_value("slope_scale"))
        except:
            pass

    if dpg.does_item_exist("rp_ip"):
        try:
            params["rp_ip"] = str(dpg.get_value("rp_ip")).strip()
        except:
            pass
    if dpg.does_item_exist("output_channel"):
        try:
            params["output_channel"] = int(dpg.get_value("output_channel"))
        except:
            pass

    if dpg.does_item_exist("global_t_hold_high"):
        try:
            global_timing["t_hold_high"] = gui_to_seconds(float(dpg.get_value("global_t_hold_high")))
        except:
            pass
    if dpg.does_item_exist("global_t_hold_mid"):
        try:
            global_timing["t_hold_mid"] = gui_to_seconds(float(dpg.get_value("global_t_hold_mid")))
        except:
            pass

    for w in (waveform1_params, waveform2_params):
        w["slope_ref_hz"] = params["slope_ref_hz"]
        w["slope_scale"] = params["slope_scale"]
        w["time_unit"] = params["time_unit"]

def pull_tab_params(tab_id):
    pull_global_params()

    target = waveform1_params if tab_id == 1 else waveform2_params

    for key in ["v_low", "v_high", "v_mid"]:
        wid = f"{key}_{tab_id}"
        if dpg.does_item_exist(wid):
            try:
                target[key] = float(dpg.get_value(wid))
            except:
                pass

    if (sweep_time_enabled or (not sweep_voltage_enabled)):
        for key in ["t_hold_high", "t_hold_mid"]:
            wid = f"{key}_{tab_id}"
            if dpg.does_item_exist(wid):
                try:
                    target[key] = gui_to_seconds(float(dpg.get_value(wid)))
                except:
                    pass

    if tab_id == 1:
        update_slope_reference_from_initial()

def update_preview(tab_id):
    pull_tab_params(tab_id)

    p = waveform1_params if tab_id == 1 else waveform2_params

    trise_s = derived_t_rise_s(p)
    if dpg.does_item_exist(f"t_rise_{tab_id}"):
        dpg.set_value(f"t_rise_{tab_id}", seconds_to_gui(trise_s))

    t, y, _, _ = generate_cycle_waveform(p, num_samples=2048)
    series = "plot1_series" if tab_id == 1 else "plot2_series"
    dpg.set_value(series, [t.tolist(), y.tolist()])

    update_points_estimate_ui()

def update_increments():
    for k in ["v_low", "v_high", "v_mid"]:
        wid = f"inc_{k}"
        if dpg.does_item_exist(wid):
            try:
                increments[k] = float(dpg.get_value(wid))
            except:
                pass

    for k in ["t_hold_high", "t_hold_mid"]:
        wid = f"inc_{k}"
        if dpg.does_item_exist(wid):
            try:
                increments[k] = gui_to_seconds(float(dpg.get_value(wid)))
            except:
                pass

    update_points_estimate_ui()

def apply_visibility_rules():
    pull_global_params()

    _show_item("global_timing_group", show=bool(sweep_voltage_enabled))

    for tab_id in (1, 2):
        _show_item(f"voltage_group_{tab_id}", show=bool(sweep_voltage_enabled))

    show_tab_time = bool(sweep_time_enabled or (not sweep_voltage_enabled))
    for tab_id in (1, 2):
        _show_item(f"time_group_{tab_id}", show=show_tab_time)

    _show_item("inc_voltage_group", show=bool(sweep_voltage_enabled))
    _show_item("inc_time_group", show=bool(sweep_time_enabled))

    update_preview(1)
    update_preview(2)
    update_points_estimate_ui()

def on_mode_toggle(sender=None, app_data=None, user_data=None):
    apply_visibility_rules()

def on_time_unit_changed(sender, app_data):
    old_unit = params["time_unit"]
    if app_data not in TIME_UNITS or app_data == old_unit:
        return

    params["time_unit"] = app_data

    for tab_id, w in [(1, waveform1_params), (2, waveform2_params)]:
        for key in ["t_hold_high", "t_hold_mid"]:
            wid = f"{key}_{tab_id}"
            if dpg.does_item_exist(wid):
                dpg.set_value(wid, seconds_to_gui(float(w[key])))

        trise_s = derived_t_rise_s(w)
        if dpg.does_item_exist(f"t_rise_{tab_id}"):
            dpg.set_value(f"t_rise_{tab_id}", seconds_to_gui(trise_s))

    if dpg.does_item_exist("global_t_hold_high"):
        dpg.set_value("global_t_hold_high", seconds_to_gui(float(global_timing["t_hold_high"])))
    if dpg.does_item_exist("global_t_hold_mid"):
        dpg.set_value("global_t_hold_mid", seconds_to_gui(float(global_timing["t_hold_mid"])))

    for k in ["t_hold_high", "t_hold_mid"]:
        wid = f"inc_{k}"
        if dpg.does_item_exist(wid):
            dpg.set_value(wid, seconds_to_gui(float(increments[k])))

    update_preview(1)
    update_preview(2)
    update_points_estimate_ui()

def start_sweep():
    pull_tab_params(1)
    pull_tab_params(2)
    update_increments()

    predicted_pts, nseg, _, _ = estimate_points_and_recommendations()
    if predicted_pts > RP_ARB_LIMIT:
        dpg.set_value(
            "status_text",
            f"Blocked: estimated {predicted_pts} pts > {RP_ARB_LIMIT}. "
            f"Increase increments or reduce sweep range."
        )
        update_points_estimate_ui()
        return

    global last_total_time
    _, _, _, total_time = generate_sweep()
    last_total_time = total_time

    # generate_sweep()
    update_preview(1)
    update_preview(2)
    deploy_sweep()


# ============================================================
# Click info / highlight
# ============================================================
def highlight_segment(x_start, x_end, y0, y1):
    if dpg.does_item_exist("highlight_box"):
        dpg.delete_item("highlight_box")
    margin = 0.02
    with dpg.draw_layer(parent="sweep_plot", tag="highlight_box"):
        dpg.draw_rectangle(
            pmin=(x_start, y0 - margin),
            pmax=(x_end, y1 + margin),
            color=(128, 0, 255, 180),
            thickness=2,
            fill=(0, 0, 0, 0),
        )

def on_left_click():
    if not dpg.is_item_hovered("sweep_plot"):
        return

    plot_x, plot_y = dpg.get_plot_mouse_pos()

    for i, (x_start, x_end, y0, y1) in enumerate(segment_bounds):
        if x_start <= plot_x <= x_end and (y0 - 0.05) <= plot_y <= (y1 + 0.05):
            seg = saved_sweep_steps[i].copy()
            seg["t_rise_s(derived)"] = derived_t_rise_s(seg)

            lines = []
            for k in sorted(seg.keys()):
                v = seg[k]
                if isinstance(v, (int, float)):
                    lines.append(f"{k}: {v:.6g}")
                else:
                    lines.append(f"{k}: {v}")
            dpg.set_value("waveform_output_text", "\n".join(lines))
            highlight_segment(x_start, x_end, y0, y1)
            return

    dpg.set_value("waveform_output_text", "Clicked outside waveform.")


# ============================================================
# Save/load
# ============================================================
def save_settings():
    filename = dpg.get_value("filename_input") or SAVE_PATH
    if not filename.endswith(".json"):
        filename += ".json"
    blob = {
        "params": params,
        "waveform1_params": waveform1_params,
        "waveform2_params": waveform2_params,
        "increments": increments,
        "sweep_voltage_enabled": bool(sweep_voltage_enabled),
        "sweep_time_enabled": bool(sweep_time_enabled),
        "global_timing": global_timing,
    }
    try:
        with open(filename, "w") as f:
            json.dump(blob, f, indent=4)
        dpg.set_value("status_text", f"Saved '{filename}'")
    except Exception as e:
        dpg.set_value("status_text", f"Save error: {e}")

def load_settings():
    global sweep_voltage_enabled, sweep_time_enabled

    filename = dpg.get_value("filename_input") or SAVE_PATH
    if not filename.endswith(".json"):
        filename += ".json"
    if not os.path.exists(filename):
        dpg.set_value("status_text", f"File not found: {filename}")
        return
    try:
        with open(filename, "r") as f:
            blob = json.load(f)

        params.update(blob.get("params", {}))
        waveform1_params.update(blob.get("waveform1_params", {}))
        waveform2_params.update(blob.get("waveform2_params", {}))
        increments.update(blob.get("increments", {}))
        global_timing.update(blob.get("global_timing", global_timing))

        sweep_voltage_enabled = bool(blob.get("sweep_voltage_enabled", True))
        sweep_time_enabled = bool(blob.get("sweep_time_enabled", True))

        dpg.set_value("rp_ip", str(params.get("rp_ip", "rp-f0cbdd.local")))
        dpg.set_value("output_channel", str(int(params.get("output_channel", 1))))
        dpg.set_value("chk_sweep_voltage", bool(sweep_voltage_enabled))
        dpg.set_value("chk_sweep_time", bool(sweep_time_enabled))

        dpg.set_value("slope_ref_hz", float(params.get("slope_ref_hz", 5000.0)))
        dpg.set_value("slope_scale", float(params.get("slope_scale", 1.0)))

        tu = params.get("time_unit", "ms")
        if tu not in TIME_UNITS:
            tu = "ms"
            params["time_unit"] = "ms"
        dpg.set_value("time_units", tu)

        dpg.set_value("global_t_hold_high", seconds_to_gui(float(global_timing["t_hold_high"])))
        dpg.set_value("global_t_hold_mid", seconds_to_gui(float(global_timing["t_hold_mid"])))

        for tab_id, w in [(1, waveform1_params), (2, waveform2_params)]:
            for key in ["v_low", "v_high", "v_mid"]:
                dpg.set_value(f"{key}_{tab_id}", float(w[key]))
            for key in ["t_hold_high", "t_hold_mid"]:
                dpg.set_value(f"{key}_{tab_id}", seconds_to_gui(float(w[key])))

        for k in ["v_low", "v_high", "v_mid"]:
            dpg.set_value(f"inc_{k}", float(increments.get(k, 0.0)))
        for k in ["t_hold_high", "t_hold_mid"]:
            dpg.set_value(f"inc_{k}", seconds_to_gui(float(increments.get(k, 0.0))))

        update_slope_reference_from_initial()
        apply_visibility_rules()
        update_points_estimate_ui()
        dpg.set_value("status_text", f"Loaded '{filename}'")
    except Exception as e:
        dpg.set_value("status_text", f"Load error: {e}")


# ============================================================
# GUI
# ============================================================
def add_waveform_inputs(tab_id, values):
    with dpg.group():
        with dpg.group(tag=f"voltage_group_{tab_id}"):
            dpg.add_text("Levels (V)")
            for key in ["v_low", "v_high", "v_mid"]:
                dpg.add_input_float(
                    label=key,
                    tag=f"{key}_{tab_id}",
                    default_value=float(values[key]),
                    width=200,
                    callback=lambda s, a, tid=tab_id: update_preview(tid),
                    format="%.6f",
                )

        dpg.add_spacer(height=6)

        with dpg.group(tag=f"time_group_{tab_id}"):
            dpg.add_text("Timing (in selected Time Units)")
            for key in ["t_hold_high", "t_hold_mid"]:
                dpg.add_input_float(
                    label=key,
                    tag=f"{key}_{tab_id}",
                    default_value=seconds_to_gui(float(values[key])),
                    width=200,
                    callback=lambda s, a, tid=tab_id: update_preview(tid),
                    format="%.6f",
                )

        dpg.add_spacer(height=6)

        dpg.add_text("Derived (read-only)")
        dpg.add_input_float(
            label="t_rise (derived)",
            tag=f"t_rise_{tab_id}",
            default_value=seconds_to_gui(float(derived_t_rise_s(values))),
            width=200,
            readonly=True,
            enabled=False,
            format="%.6f",
        )

def add_increment_inputs():
    with dpg.group():
        with dpg.group(tag="inc_voltage_group"):
            dpg.add_text("Voltage increments")
            for k in ["v_low", "v_high", "v_mid"]:
                dpg.add_input_float(
                    label=f"inc_{k}",
                    tag=f"inc_{k}",
                    default_value=float(increments.get(k, 0.0)),
                    width=200,
                    callback=update_increments,
                    format="%.6f",
                )

        dpg.add_spacer(height=8)

        with dpg.group(tag="inc_time_group"):
            dpg.add_text("Time increments (in selected Time Units)")
            for k in ["t_hold_high", "t_hold_mid"]:
                dpg.add_input_float(
                    label=f"inc_{k}",
                    tag=f"inc_{k}",
                    default_value=seconds_to_gui(float(increments.get(k, 0.0))),
                    width=200,
                    callback=update_increments,
                    format="%.6f",
                )

def build_gui():
    dpg.create_context()

    with dpg.window(label="RP Sweep GUI (fixed units + fixed slope + stable sender)", width=1120, height=1030):
        dpg.add_input_text(label="Save/Load Filename", tag="filename_input", default_value=SAVE_PATH, width=360)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Save", callback=save_settings)
            dpg.add_button(label="Load", callback=load_settings)

        dpg.add_separator()

        dpg.add_input_text(label="Red Pitaya IP", tag="rp_ip", default_value=str(params["rp_ip"]), width=260)
        dpg.add_combo(label="Output Channel", items=["1", "2"], default_value="1", tag="output_channel", width=90)

        dpg.add_separator()

        dpg.add_combo(
            label="Time Units",
            tag="time_units",
            items=TIME_UNIT_ORDER,
            default_value=params["time_unit"],
            callback=on_time_unit_changed,
            width=100,
        )

        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="Sweep Voltage", tag="chk_sweep_voltage", default_value=True, callback=on_mode_toggle)
            dpg.add_checkbox(label="Sweep Time", tag="chk_sweep_time", default_value=True, callback=on_mode_toggle)

        dpg.add_separator()

        with dpg.group(tag="global_timing_group"):
            dpg.add_text("Global Timing (used when sweeping voltage ONLY; keeps tabs uncluttered)")
            dpg.add_input_float(
                label="t_hold_high",
                tag="global_t_hold_high",
                default_value=seconds_to_gui(float(global_timing["t_hold_high"])),
                width=220,
                callback=lambda s, a: (pull_global_params(), update_preview(1), update_preview(2)),
                format="%.6f",
            )
            dpg.add_input_float(
                label="t_hold_mid",
                tag="global_t_hold_mid",
                default_value=seconds_to_gui(float(global_timing["t_hold_mid"])),
                width=220,
                callback=lambda s, a: (pull_global_params(), update_preview(1), update_preview(2)),
                format="%.6f",
            )

        dpg.add_separator()

        dpg.add_text("Slope reference (defines ramp slope only):")
        dpg.add_input_float(
            label="slope_ref_hz (Hz)",
            tag="slope_ref_hz",
            default_value=float(params.get("slope_ref_hz", 5000.0)),
            width=220,
            callback=lambda s, a: (pull_global_params(), update_slope_reference_from_initial(), update_preview(1), update_preview(2)),
            format="%.6f",
        )
        dpg.add_input_float(
            label="slope_scale",
            tag="slope_scale",
            default_value=float(params.get("slope_scale", 1.0)),
            width=220,
            callback=lambda s, a: (pull_global_params(), update_slope_reference_from_initial(), update_preview(1), update_preview(2)),
            format="%.6f",
        )

        dpg.add_separator()
        dpg.add_button(label="Start Sweep (Generate + Send)", callback=start_sweep)
        dpg.add_text("", tag="status_text")

        dpg.add_separator()
        dpg.add_text("Sweep increments:")
        add_increment_inputs()

        dpg.add_separator()
        dpg.add_text("", tag="points_estimate_text")

        dpg.add_separator()

        with dpg.tab_bar():
            with dpg.tab(label="Initial Waveform"):
                add_waveform_inputs(1, waveform1_params)
                with dpg.plot(label="Initial Preview", height=240, width=980):
                    dpg.add_plot_axis(dpg.mvXAxis)
                    dpg.add_plot_axis(dpg.mvYAxis, tag="axis1")
                    dpg.add_line_series([], [], parent="axis1", tag="plot1_series")
                    dpg.set_axis_limits("axis1", y_min, y_max)

            with dpg.tab(label="Final Waveform"):
                add_waveform_inputs(2, waveform2_params)
                with dpg.plot(label="Final Preview", height=240, width=980):
                    dpg.add_plot_axis(dpg.mvXAxis)
                    dpg.add_plot_axis(dpg.mvYAxis, tag="axis2")
                    dpg.add_line_series([], [], parent="axis2", tag="plot2_series")
                    dpg.set_axis_limits("axis2", y_min, y_max)

            with dpg.tab(label="Sweep Visual"):
                with dpg.plot(label="Sweep Preview", height=340, width=980, tag="sweep_plot"):
                    dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_sweep")
                    dpg.add_plot_axis(dpg.mvYAxis, tag="axis_sweep")
                    dpg.add_line_series([], [], parent="axis_sweep", tag="sweep_plot_series")
                    dpg.set_axis_limits("axis_sweep", y_min, y_max)

                dpg.add_text("Clicked Segment Params:")
                dpg.add_text("", tag="waveform_output_text")

                with dpg.handler_registry():
                    dpg.add_mouse_click_handler(callback=lambda: on_left_click())

    dpg.create_viewport(title="RP Sweep GUI", width=1180, height=1080)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    pull_global_params()
    update_slope_reference_from_initial()
    apply_visibility_rules()
    update_points_estimate_ui()

    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    build_gui()
