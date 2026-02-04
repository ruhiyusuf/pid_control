import numpy as np
import dearpygui.dearpygui as dpg
import socket, struct
import json, os
import time
import threading

# ============================================================
# Red Pitaya protocol
# Packet format:
#   [uint32 nbytes][uint8 channel][float32 duration_s][int16 payload]
# Header fields BIG-endian. Payload raw int16 bytes.
# ============================================================

SAVE_PATH = "waveform_settings.json"

TIME_UNITS = {"us": 1e-6, "ms": 1e-3, "s": 1.0}
TIME_UNIT_ORDER = ["us", "ms", "s"]

# ---------------------------
# Parameters (internal timing is seconds)
# ---------------------------
params = {
    "rp_ip": "rp-f0cbdd.local",
    "rp_port": 9000,
    "output_channel": 1,

    # Slope reference (only used to compute ramp slope)
    "slope_ref_hz": 5000.0,
    "slope_scale": 1.0,

    # waveform levels (V)
    "v_low": -0.2,
    "v_high": 1.0,
    "v_mid": 0.0,

    # timing (seconds)
    "t_hold_high": 2e-3,
    "t_hold_mid": 2e-3,

    # UI time unit (display only)
    "time_unit": "ms",
}

waveform1_params = params.copy()  # initial
waveform2_params = params.copy()  # final

# sweep mode toggles
sweep_voltage_enabled = True
sweep_time_enabled = True

# global timing when sweeping voltage only
global_timing = {
    "t_hold_high": float(params["t_hold_high"]),
    "t_hold_mid": float(params["t_hold_mid"]),
}

# increments (timing increments stored as seconds)
increments = {
    "v_low": 0.02,
    "v_high": 0.02,
    "v_mid": 0.02,
    "t_hold_high": 0.2e-3,
    "t_hold_mid": 0.2e-3,
}

# plotting limits
y_min, y_max = -0.5, 1.2

# sampling
SAMPLES_PER_SEG = 2048
RP_ARB_LIMIT = 16348

# streaming behavior
REPEAT_CYCLES_PER_STEP = 3
EXTRA_DWELL_BETWEEN_STEPS_S = 0.0  # optional extra pause between steps

# one background worker thread
_worker_thread = None
_stop_event = threading.Event()

# cached slope reference (V/s) computed from initial waveform1
SLOPE_REF_V_PER_S = None

VOLT_KEYS = {"v_low", "v_high", "v_mid"}
TIME_KEYS = {"t_hold_high", "t_hold_mid"}

# --- Trigger output (RP CH2) ---
TRIG_ENABLED = True
TRIG_MODE = "gate"   # "pulse_start", "pulse_end", "gate"
TRIG_CHANNEL = 2            # use RP channel 2 for trigger
TRIG_V_HIGH = 1.0           # volts, mapped by volts_to_i16_fixed
TRIG_V_LOW = 0.0
TRIG_PULSE_WIDTH_S = 2e-3   # 2 ms pulse
TRIG_SAMPLES = 1024

# ============================================================
# Unit conversion helpers (UI <-> seconds)
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
# Slope reference + derived rise time (seconds)
# ============================================================

def triangle_slope_from_freq(v_low, v_high, freq_hz, slope_scale=1.0):
    dv = float(v_high) - float(v_low)
    # triangle: slope magnitude = 2*dv*f
    return abs(2.0 * dv * float(freq_hz)) * float(slope_scale)

def compute_rise_time_s_from_slope(v_start, v_end, slope_v_per_s):
    dv = abs(float(v_end) - float(v_start))
    if slope_v_per_s <= 1e-12:
        return 0.0
    return dv / float(slope_v_per_s)

def update_slope_reference_from_initial():
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
    if SLOPE_REF_V_PER_S is None:
        update_slope_reference_from_initial()
    slope = float(SLOPE_REF_V_PER_S or 0.0)
    return compute_rise_time_s_from_slope(p["v_low"], p["v_high"], slope)


# ============================================================
# Waveform generation (one cycle)
# ramp up -> hold high -> instant drop -> hold mid
# ============================================================

def _merge_globals_into_params(p):
    q = p.copy()
    if sweep_voltage_enabled and (not sweep_time_enabled):
        q["t_hold_high"] = float(global_timing["t_hold_high"])
        q["t_hold_mid"]  = float(global_timing["t_hold_mid"])
    return q

def generate_cycle_waveform(p, num_samples=2048):
    p = _merge_globals_into_params(p)

    v_low  = float(p["v_low"])
    v_high = float(p["v_high"])
    v_mid  = float(p["v_mid"])

    t_rise = max(0.0, derived_t_rise_s(p))
    t_hh   = max(0.0, float(p["t_hold_high"]))
    t_hm   = max(0.0, float(p["t_hold_mid"]))

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

    # hold mid
    hm_mask = t >= t2
    y[hm_mask] = v_mid

    return t, y, total_time

def _seg_duration_from_t(t):
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        return float(t[-1]) if len(t) else 0.0
    dt = float(t[1] - t[0])
    return float(t[-1] + dt)


# ============================================================
# RP scaling + send
# ============================================================

def volts_to_i16_minmax(v):
    
    w = np.asarray(v, dtype=float)
    vmin = float(np.min(w))
    vmax = float(np.max(w))
    if vmax - vmin < 1e-9:
        vmax = vmin + 1e-9
    w_norm = 2.0 * (w - vmin) / (vmax - vmin) - 1.0
    w_norm = np.clip(w_norm, -1.0, 1.0)
    return (w_norm * (2**13 - 1)).astype(np.int16), vmin, vmax

RP_DAC_BITS = 14
RP_DAC_MAX = (2**(RP_DAC_BITS-1) - 1)  # 8191 for 14-bit signed
# NOTE: your code used (2**13-1)=8191 already; keep consistent

def volts_to_i16_fixed(v, v_fullscale=1.0, invert=False):
    """
    Map volts directly to DAC codes using a fixed full-scale voltage.
    v_fullscale=1.0 means +1.0V maps to +8191, -1.0V maps to -8191.
    """
    w = np.asarray(v, dtype=float)
    w_norm = w / float(v_fullscale)      # fixed mapping
    w_norm = np.clip(w_norm, -1.0, 1.0)  # prevent overflow

    if invert:
        w_norm = -w_norm

    return (w_norm * RP_DAC_MAX).astype(np.int16)

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

def deploy_one_cycle(p):
    rp_ip = str(dpg.get_value("rp_ip")).strip()
    rp_port = int(params["rp_port"])
    ch = int(dpg.get_value("output_channel"))

    t, y, _ = generate_cycle_waveform(p, num_samples=SAMPLES_PER_SEG)
    duration_cycle = _seg_duration_from_t(t)

    if SAMPLES_PER_SEG > RP_ARB_LIMIT:
        raise ValueError(f"SAMPLES_PER_SEG={SAMPLES_PER_SEG} > RP_ARB_LIMIT={RP_ARB_LIMIT}")

    # wave_i16, vmin, vmax = volts_to_i16_minmax(y)
    wave_i16 = volts_to_i16_fixed(y, v_fullscale=1.0, invert=False)
    vmin, vmax = float(np.min(y)), float(np.max(y))  # just for status display
    rp_send_waveform_big_endian(wave_i16, duration_cycle, rp_ip, rp_port, ch)
    return duration_cycle, vmin, vmax


# ============================================================
# Build sweep steps list (THIS is the sweep logic)
# ============================================================

def _build_param_steps_list():
    """
    Returns list of param dicts for each step (includes step 0 = initial).
    """
    active_keys = _active_sweep_keys()
    interp = waveform1_params.copy()
    steps = [ _merge_globals_into_params(interp) ]

    guard = 200000
    for _ in range(guard):
        done = True
        for key in active_keys:
            v1 = waveform1_params.get(key, None)
            v2 = waveform2_params.get(key, None)
            if v1 is None or v2 is None:
                continue
            if (not _is_number(v1)) or (not _is_number(v2)):
                continue

            inc = float(increments.get(key, 0.0))
            curr = float(interp.get(key, v1))
            target = float(v2)

            new_val = _step_toward(curr, target, inc)
            interp[key] = new_val
            if abs(new_val - target) > 1e-12:
                done = False

        steps.append(_merge_globals_into_params(interp))

        if done:
            break
    else:
        raise RuntimeError("Sweep exceeded guard; increments too small?")

    return steps


# ============================================================
# Sweep time estimation + UI
# ============================================================

def _fmt_time(seconds: float) -> str:
    seconds = float(max(0.0, seconds))
    if seconds < 1.0:
        return f"{seconds*1e3:.3g} ms"
    if seconds < 60.0:
        return f"{seconds:.3g} s"
    m = int(seconds // 60)
    s = seconds - 60*m
    return f"{m} min {s:.1f} s"

def estimate_total_sweep_time_s():
    update_slope_reference_from_initial()
    steps = _build_param_steps_list()

    total = 0.0
    for p in steps:
        t, _, _ = generate_cycle_waveform(p, num_samples=512)
        dur = _seg_duration_from_t(t)
        total += dur * float(max(1, REPEAT_CYCLES_PER_STEP))
        total += float(max(0.0, EXTRA_DWELL_BETWEEN_STEPS_S))
    return float(total), len(steps)

def update_sweep_time_ui():
    if not dpg.does_item_exist("sweep_time_text"):
        return
    try:
        total_s, nsteps = estimate_total_sweep_time_s()
        dpg.set_value(
            "sweep_time_text",
            f"Estimated sweep time: {_fmt_time(total_s)}\n"
            f"Steps: {nsteps} (includes initial)\n"
            f"Repeat cycles per step: {int(max(1, REPEAT_CYCLES_PER_STEP))}\n"
            f"Extra dwell between steps: {EXTRA_DWELL_BETWEEN_STEPS_S:.3g} s\n"
            f"Samples per cycle sent: {SAMPLES_PER_SEG} / {RP_ARB_LIMIT}"
        )
    except Exception as e:
        dpg.set_value("sweep_time_text", f"Estimated sweep time: (error) {e}")


# ============================================================
# Sweep preview generation (for visualization only)
# ============================================================

def generate_sweep_preview_for_plot(max_plot_points=12000):
    """
    Builds a full preview of the *stepwise* sweep:
    each step's cycle waveform repeated REPEAT_CYCLES_PER_STEP times,
    plus optional EXTRA_DWELL_BETWEEN_STEPS_S as flat hold at last sample.

    For plotting, we may downsample to max_plot_points.
    """
    update_slope_reference_from_initial()
    steps = _build_param_steps_list()

    xs = []
    ys = []
    t_cursor = 0.0

    for p in steps:
        t, y, _ = generate_cycle_waveform(p, num_samples=512)  # lighter for preview
        seg_dur = _seg_duration_from_t(t)

        for _ in range(int(max(1, REPEAT_CYCLES_PER_STEP))):
            xs.append(t + t_cursor)
            ys.append(y.copy())
            t_cursor += seg_dur

        if EXTRA_DWELL_BETWEEN_STEPS_S > 0:
            # create a small flat segment (10 pts)
            dwell_n = 10
            tt = np.linspace(0, EXTRA_DWELL_BETWEEN_STEPS_S, dwell_n, endpoint=False)
            yy = np.full(dwell_n, float(y[-1]))
            xs.append(tt + t_cursor)
            ys.append(yy)
            t_cursor += float(EXTRA_DWELL_BETWEEN_STEPS_S)

    if not xs:
        return np.array([]), np.array([])

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    # downsample for plotting speed
    if len(x) > max_plot_points:
        idx = np.linspace(0, len(x)-1, max_plot_points).astype(int)
        x = x[idx]
        y = y[idx]

    return x, y

def update_sweep_preview_ui():
    if not dpg.does_item_exist("sweep_plot_series"):
        return
    try:
        x, y = generate_sweep_preview_for_plot()
        dpg.set_value("sweep_plot_series", [x, y])
        dpg.set_value("sweep_preview_status", f"Sweep preview points: {len(x)}")
    except Exception as e:
        dpg.set_value("sweep_preview_status", f"Sweep preview error: {e}")


# ============================================================
# Interruptible sleep (so Stop works immediately)
# ============================================================

def _interruptible_sleep(total_s: float, poll_s: float = 0.01) -> bool:
    t0 = time.time()
    while (time.time() - t0) < float(total_s):
        if _stop_event.is_set():
            return False
        time.sleep(poll_s)
    return True

def _send_trigger_waveform(levels, duration_s):
    """Send a simple waveform on TRIG_CHANNEL."""
    rp_ip = str(dpg.get_value("rp_ip")).strip()
    rp_port = int(params["rp_port"])
    wave_i16 = volts_to_i16_fixed(np.array(levels, dtype=float), v_fullscale=1.0, invert=False)
    rp_send_waveform_big_endian(wave_i16, float(duration_s), rp_ip, rp_port, int(TRIG_CHANNEL))

def send_trigger_once_per_sweep(total_sweep_s: float, pulse_width_s: float = 2e-3):
    """
    Creates a waveform that is low almost everywhere, with ONE pulse near the start.
    Because the RP output repeats the buffer, it will repeat only once per total_sweep_s.
    """
    if not TRIG_ENABLED:
        return

    total_sweep_s = float(max(total_sweep_s, pulse_width_s * 5))
    pulse_width_s = float(max(1e-6, pulse_width_s))

    n = int(max(256, TRIG_SAMPLES))  # keep modest
    levels = np.full(n, TRIG_V_LOW, dtype=float)

    # pulse width in samples
    pulse_n = max(1, int(round(pulse_width_s / total_sweep_s * n)))
    pulse_n = min(pulse_n, n // 4)

    # put pulse very early (starting at sample 2)
    start = 2
    end = min(n, start + pulse_n)
    levels[start:end] = TRIG_V_HIGH

    _send_trigger_waveform(levels, total_sweep_s)

def send_trigger_gate(high: bool, gate_duration_s: float = 5e-3):
    """Gate high/low (useful if you want a 'sweep active' indicator)."""
    if not TRIG_ENABLED:
        return
    n = int(max(64, TRIG_SAMPLES))
    level = TRIG_V_HIGH if high else TRIG_V_LOW
    levels = np.full(n, level, dtype=float)
    _send_trigger_waveform(levels, float(gate_duration_s))


# ============================================================
# One worker thread: does the whole sweep (stepwise)
# ============================================================

def _sweep_worker():
    try:
        _stop_event.clear()
        update_slope_reference_from_initial()

        steps = _build_param_steps_list()
        total_s_est, _ = estimate_total_sweep_time_s()

        dpg.set_value("status_text", f"Running sweep... total est {_fmt_time(total_s_est)}")
        start_time = time.time()

        # ---- trigger at sweep start ----
        if TRIG_ENABLED:
            if TRIG_MODE == "pulse_start":
                total_s_est, _ = estimate_total_sweep_time_s()

                # one pulse per sweep duration
                send_trigger_once_per_sweep(total_s_est, pulse_width_s=TRIG_PULSE_WIDTH_S)

            elif TRIG_MODE == "gate":
                # make gate high briefly; if you want it held the whole sweep, see note below
                send_trigger_gate(True, gate_duration_s=TRIG_PULSE_WIDTH_S)

        for idx, p in enumerate(steps, start=1):
            if _stop_event.is_set():
                break

            # send this step
            dur_cycle, vmin, vmax = deploy_one_cycle(p)
            wait_s = float(max(1, REPEAT_CYCLES_PER_STEP)) * float(dur_cycle)

            # live step display (so you can confirm it is changing)
            if dpg.does_item_exist("current_step_text"):
                dpg.set_value(
                    "current_step_text",
                    f"Step {idx}/{len(steps)}\n"
                    f"v_low={p['v_low']:.4g}, v_high={p['v_high']:.4g}, v_mid={p['v_mid']:.4g}\n"
                    f"t_hold_high={p['t_hold_high']:.4g}s, t_hold_mid={p['t_hold_mid']:.4g}s\n"
                    f"t_rise={derived_t_rise_s(p):.4g}s"
                )

            # wait for repeats to finish playing
            if not _interruptible_sleep(wait_s, poll_s=0.01):
                break

            # optional extra dwell (makes step change obvious)
            if EXTRA_DWELL_BETWEEN_STEPS_S > 0:
                if not _interruptible_sleep(EXTRA_DWELL_BETWEEN_STEPS_S, poll_s=0.01):
                    break

            elapsed = time.time() - start_time
            dpg.set_value(
                "status_text",
                f"Step {idx}/{len(steps)} done | cycle={dur_cycle:.6g}s | repeats={int(max(1, REPEAT_CYCLES_PER_STEP))} "
                f"| elapsed {_fmt_time(elapsed)} | mapped [{vmin:.3f},{vmax:.3f}]"
            )
        # ---- trigger at sweep end ----
        if TRIG_ENABLED and (not _stop_event.is_set()):
            if TRIG_MODE == "pulse_end":
                send_trigger_pulse()
            elif TRIG_MODE == "gate":
                send_trigger_gate(False, gate_duration_s=TRIG_PULSE_WIDTH_S)

        if _stop_event.is_set():
            dpg.set_value("status_text", "Stopped.")
        else:
            dpg.set_value("status_text", "Finished.")

    except Exception as e:
        dpg.set_value("status_text", f"ERROR: {e}")


# ============================================================
# Control callbacks
# ============================================================

def start_sweep():
    global _worker_thread, REPEAT_CYCLES_PER_STEP, SAMPLES_PER_SEG, EXTRA_DWELL_BETWEEN_STEPS_S

    pull_tab_params(1)
    pull_tab_params(2)
    update_increments()

    # read repeats + samples + dwell from UI
    if dpg.does_item_exist("repeat_cycles_ui"):
        try:
            REPEAT_CYCLES_PER_STEP = int(max(1, dpg.get_value("repeat_cycles_ui")))
        except:
            pass
    if dpg.does_item_exist("samples_per_seg_ui"):
        try:
            SAMPLES_PER_SEG = int(max(256, dpg.get_value("samples_per_seg_ui")))
        except:
            pass
    if dpg.does_item_exist("extra_dwell_ui"):
        try:
            EXTRA_DWELL_BETWEEN_STEPS_S = float(max(0.0, dpg.get_value("extra_dwell_ui")))
        except:
            pass

    update_slope_reference_from_initial()
    update_preview(1)
    update_preview(2)
    update_sweep_time_ui()
    update_sweep_preview_ui()

    if SAMPLES_PER_SEG > RP_ARB_LIMIT:
        dpg.set_value("status_text", f"Blocked: samples per cycle {SAMPLES_PER_SEG} > limit {RP_ARB_LIMIT}")
        return

    # quick sanity check: do we actually have more than 1 step?
    try:
        steps = _build_param_steps_list()
        if len(steps) <= 1:
            dpg.set_value("status_text", "No sweep steps generated (initial == final or increment is 0).")
            return
    except Exception as e:
        dpg.set_value("status_text", f"Sweep build error: {e}")
        return

    if _worker_thread is not None and _worker_thread.is_alive():
        dpg.set_value("status_text", "Already running.")
        return

    _stop_event.clear()
    _worker_thread = threading.Thread(target=_sweep_worker, daemon=True)
    _worker_thread.start()

def stop_sweep():
    _stop_event.set()

def regenerate_sweep_preview():
    pull_tab_params(1)
    pull_tab_params(2)
    update_increments()
    update_sweep_time_ui()
    update_sweep_preview_ui()


# ============================================================
# GUI sync + visibility logic
# ============================================================

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

def update_preview(tab_id):
    pull_tab_params(tab_id)
    p = waveform1_params if tab_id == 1 else waveform2_params

    trise_s = derived_t_rise_s(p)
    if dpg.does_item_exist(f"t_rise_{tab_id}"):
        dpg.set_value(f"t_rise_{tab_id}", seconds_to_gui(trise_s))

    t, y, _ = generate_cycle_waveform(p, num_samples=2048)
    series = "plot1_series" if tab_id == 1 else "plot2_series"
    dpg.set_value(series, [t.tolist(), y.tolist()])

def _show_item(tag, show=True):
    if dpg.does_item_exist(tag):
        dpg.configure_item(tag, show=show)

def apply_visibility_rules():
    pull_global_params()

    _show_item("global_timing_group", show=bool(sweep_voltage_enabled))
    show_tab_time = bool(sweep_time_enabled or (not sweep_voltage_enabled))
    for tab_id in (1, 2):
        _show_item(f"time_group_{tab_id}", show=show_tab_time)

    _show_item("inc_voltage_group", show=bool(sweep_voltage_enabled))
    _show_item("inc_time_group", show=bool(sweep_time_enabled))

    update_preview(1)
    update_preview(2)
    update_sweep_time_ui()
    update_sweep_preview_ui()

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
    update_sweep_time_ui()
    update_sweep_preview_ui()


# ============================================================
# Save / load
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
        "REPEAT_CYCLES_PER_STEP": int(max(1, REPEAT_CYCLES_PER_STEP)),
        "SAMPLES_PER_SEG": int(SAMPLES_PER_SEG),
        "EXTRA_DWELL_BETWEEN_STEPS_S": float(max(0.0, EXTRA_DWELL_BETWEEN_STEPS_S)),
    }
    try:
        with open(filename, "w") as f:
            json.dump(blob, f, indent=4)
        dpg.set_value("status_text", f"Saved '{filename}'")
    except Exception as e:
        dpg.set_value("status_text", f"Save error: {e}")

def load_settings():
    global sweep_voltage_enabled, sweep_time_enabled
    global REPEAT_CYCLES_PER_STEP, SAMPLES_PER_SEG, EXTRA_DWELL_BETWEEN_STEPS_S

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

        REPEAT_CYCLES_PER_STEP = int(max(1, blob.get("REPEAT_CYCLES_PER_STEP", REPEAT_CYCLES_PER_STEP)))
        SAMPLES_PER_SEG = int(max(256, blob.get("SAMPLES_PER_SEG", SAMPLES_PER_SEG)))
        EXTRA_DWELL_BETWEEN_STEPS_S = float(max(0.0, blob.get("EXTRA_DWELL_BETWEEN_STEPS_S", EXTRA_DWELL_BETWEEN_STEPS_S)))

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

        dpg.set_value("repeat_cycles_ui", int(REPEAT_CYCLES_PER_STEP))
        dpg.set_value("samples_per_seg_ui", int(SAMPLES_PER_SEG))
        dpg.set_value("extra_dwell_ui", float(EXTRA_DWELL_BETWEEN_STEPS_S))

        update_slope_reference_from_initial()
        apply_visibility_rules()
        dpg.set_value("status_text", f"Loaded '{filename}'")
    except Exception as e:
        dpg.set_value("status_text", f"Load error: {e}")


# ============================================================
# GUI builders
# ============================================================

def add_waveform_inputs(tab_id, values):
    with dpg.group():
        dpg.add_text("Levels (V)")
        for key in ["v_low", "v_high", "v_mid"]:
            dpg.add_input_float(
                label=key,
                tag=f"{key}_{tab_id}",
                default_value=float(values[key]),
                width=200,
                callback=lambda s, a, tid=tab_id: (update_preview(tid), update_sweep_time_ui(), update_sweep_preview_ui()),
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
                    callback=lambda s, a, tid=tab_id: (update_preview(tid), update_sweep_time_ui(), update_sweep_preview_ui()),
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
                    callback=lambda: (update_increments(), update_sweep_time_ui(), update_sweep_preview_ui()),
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
                    callback=lambda: (update_increments(), update_sweep_time_ui(), update_sweep_preview_ui()),
                    format="%.6f",
                )

def build_gui():
    dpg.create_context()

    with dpg.window(label="RP Sweep GUI (step-by-step deploy + sweep preview)", width=1220, height=1180):
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
            dpg.add_text("Global Timing (used when sweeping voltage ONLY)")
            dpg.add_input_float(
                label="t_hold_high",
                tag="global_t_hold_high",
                default_value=seconds_to_gui(float(global_timing["t_hold_high"])),
                width=220,
                callback=lambda s, a: (pull_global_params(), update_preview(1), update_preview(2), update_sweep_time_ui(), update_sweep_preview_ui()),
                format="%.6f",
            )
            dpg.add_input_float(
                label="t_hold_mid",
                tag="global_t_hold_mid",
                default_value=seconds_to_gui(float(global_timing["t_hold_mid"])),
                width=220,
                callback=lambda s, a: (pull_global_params(), update_preview(1), update_preview(2), update_sweep_time_ui(), update_sweep_preview_ui()),
                format="%.6f",
            )

        dpg.add_separator()

        dpg.add_text("Slope reference (defines ramp slope only):")
        dpg.add_input_float(
            label="slope_ref_hz (Hz)",
            tag="slope_ref_hz",
            default_value=float(params.get("slope_ref_hz", 5000.0)),
            width=220,
            callback=lambda s, a: (pull_global_params(), update_slope_reference_from_initial(), update_preview(1), update_preview(2), update_sweep_time_ui(), update_sweep_preview_ui()),
            format="%.6f",
        )
        dpg.add_input_float(
            label="slope_scale",
            tag="slope_scale",
            default_value=float(params.get("slope_scale", 1.0)),
            width=220,
            callback=lambda s, a: (pull_global_params(), update_slope_reference_from_initial(), update_preview(1), update_preview(2), update_sweep_time_ui(), update_sweep_preview_ui()),
            format="%.6f",
        )

        dpg.add_separator()
        dpg.add_text("Step deploy settings")
        with dpg.group(horizontal=True):
            dpg.add_input_int(label="Repeat cycles per step", tag="repeat_cycles_ui",
                              default_value=int(REPEAT_CYCLES_PER_STEP), width=130,
                              callback=lambda s, a: (update_sweep_time_ui(), update_sweep_preview_ui()))
            dpg.add_input_int(label="Samples per cycle", tag="samples_per_seg_ui",
                              default_value=int(SAMPLES_PER_SEG), width=130,
                              callback=lambda s, a: (update_sweep_time_ui(), update_sweep_preview_ui()))
            dpg.add_input_float(label="Extra dwell between steps (s)", tag="extra_dwell_ui",
                                default_value=float(EXTRA_DWELL_BETWEEN_STEPS_S), width=170, format="%.3f",
                                callback=lambda s, a: (update_sweep_time_ui(), update_sweep_preview_ui()))

        dpg.add_text("", tag="sweep_time_text")

        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_button(label="Regenerate Sweep Preview", callback=regenerate_sweep_preview)
            dpg.add_button(label="Start Sweep", callback=start_sweep)
            dpg.add_button(label="Stop", callback=stop_sweep)

        dpg.add_text("", tag="status_text")
        dpg.add_text("", tag="current_step_text")

        dpg.add_separator()
        dpg.add_text("Sweep increments:")
        add_increment_inputs()

        dpg.add_separator()

        with dpg.tab_bar():
            with dpg.tab(label="Initial Waveform"):
                add_waveform_inputs(1, waveform1_params)
                with dpg.plot(label="Initial Preview", height=240, width=1080):
                    dpg.add_plot_axis(dpg.mvXAxis)
                    dpg.add_plot_axis(dpg.mvYAxis, tag="axis1")
                    dpg.add_line_series([], [], parent="axis1", tag="plot1_series")
                    dpg.set_axis_limits("axis1", y_min, y_max)

            with dpg.tab(label="Final Waveform"):
                add_waveform_inputs(2, waveform2_params)
                with dpg.plot(label="Final Preview", height=240, width=1080):
                    dpg.add_plot_axis(dpg.mvXAxis)
                    dpg.add_plot_axis(dpg.mvYAxis, tag="axis2")
                    dpg.add_line_series([], [], parent="axis2", tag="plot2_series")
                    dpg.set_axis_limits("axis2", y_min, y_max)

            with dpg.tab(label="Sweep Preview"):
                dpg.add_text("This shows the *stepwise* sweep (what you are actually doing on the RP).")
                dpg.add_text("", tag="sweep_preview_status")
                with dpg.plot(label="Sweep Preview (downsampled)", height=360, width=1080):
                    dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_sweep")
                    dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_sweep")
                    dpg.add_line_series([], [], parent="y_axis_sweep", tag="sweep_plot_series")
                    dpg.set_axis_limits("y_axis_sweep", y_min, y_max)

    dpg.create_viewport(title="RP Sweep GUI", width=1260, height=1220)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    pull_global_params()
    update_slope_reference_from_initial()
    apply_visibility_rules()
    update_sweep_time_ui()
    update_sweep_preview_ui()

    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    build_gui()
