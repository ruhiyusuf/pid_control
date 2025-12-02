import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import dearpygui.dearpygui as dpg
from scipy.interpolate import interp1d
import socket, struct
import pyvisa

# ===========================
# VISA Initialization
# ===========================

rm = pyvisa.ResourceManager()

awg_connected = False

def find_awg():
    """Search available VISA resources and auto-detect Keysight/Agilent AWG."""
    devices = rm.list_resources()
    for dev in devices:
        try:
            inst = rm.open_resource(dev)
            idn = inst.query("*IDN?")
            if any(keyword in idn for keyword in ["AGILENT", "KEYSIGHT", "332", "335", "336"]):
                return dev
        except:
            continue
    return None

AWG_RESOURCE = find_awg()
if AWG_RESOURCE is None:
    print("No AWG detected! Please enter VISA address manually.")
    AWG_RESOURCE = "USB0::ENTER::MANUALLY::INSTR"  # fallback

awg = rm.open_resource(AWG_RESOURCE)
awg.write_termination = "\n"
awg.read_termination = "\n"

print("Connected to AWG:", awg.query("*IDN?"))

# === Global Parameters ===


params = {
    "rp_ip": 123,
    "v_up": 4.0,
    "v_down": 0.2,
    "baseline": 0.2,

    "t_flat_before_ms": 0.0,
    "t_ramp_ms": 1.0,
    "t_flat_after_ms": 1.0,

    "initial_freq_hz": 5000.0,  # ⭐ FIXED ⭐

    "awg_vpp": 1.0,
    "awg_offset": 0.0,
    "output_channel": 1,
}
waveform1_params = params.copy()
waveform2_params = params.copy()

waveform2_params.pop("initial_freq_hz", None)
# increments for voltages and timings
increments = {
    "v_up": 0.05,
    # "v_down": 0.05,
    "baseline": 0.05,
}

sweep_keys = list(increments.keys())
# need to change
STEPS = 6
samples_per_step = 2048 
y_min, y_max = -0.2, 1.2
sweep_running = False
sweep_paused = False
segments = []
segment_times = []
saved_sweep_steps = []
segment_bounds = []
x_vals = []
x_vals_resampled = []
y_vals_resampled = []

def reconnect_awg(address):
    """
    Try reconnecting to the AWG at the specified VISA resource string.
    (e.g., 'USB0::0x0957::0x0407::MY44019290::INSTR')
    """
    global awg, awg_connected

    try:
        if address is None or address.strip() == "":
            dpg.set_value("status_text", "No VISA address provided.")
            awg_connected = False
            return

        # Close previous connection if any
        try:
            if awg is not None:
                awg.close()
        except:
            pass

        # Open new connection
        awg = rm.open_resource(address)
        awg.write_termination = "\n"
        awg.read_termination = "\n"
        awg.timeout = 5000

        idn = awg.query("*IDN?").strip()
        awg_connected = True
        dpg.set_value("status_text", f"Reconnected: {idn}")

    except Exception as e:
        awg_connected = False
        dpg.set_value("status_text", f"Reconnect failed: {e}")

def auto_check_awg():
    global awg_connected
    address = dpg.get_value("awg_addr")

    try:
        if awg is not None:
            _ = awg.query("*OPC?")
            if not awg_connected:
                awg_connected = True
                dpg.set_value("status_text", "AWG reconnected.")
        else:
            reconnect_awg(address)

    except:
        awg_connected = False
        dpg.set_value("status_text", "AWG disconnected.")
# === Waveform Generator ===
def generate_waveform(p, num_samples=16384):
    s = p["start_v_ms"] / 1000
    u = p["slope_up_ms"] / 1000
    e = p["end_v_ms"] / 1000
    total_time = s + u + e
    t = np.linspace(0, total_time, num_samples)

    waveform = np.piecewise(t,
        [t < s,
         (t >= s) & (t < s + u),
         t >= s + u],
        [lambda t: p["v_start"],
         lambda t: p["v_start"] + (p["v_end"] - p["v_start"]) * (t - s) / u,
         lambda t: p["v_end"]])
    return t, waveform, total_time


# --- SLOPE HELPERS ---

def triangle_slope_from_freq(v_low, v_high, freq_hz):
    dv = float(v_high) - float(v_low)
    return abs(2.0 * dv * float(freq_hz))  # volts per second


def compute_ramp_time_from_slope(v_start, v_end, slope_v_per_s):
    dv = abs(float(v_start) - float(v_end))
    if slope_v_per_s < 1e-12:
        return 0.0
    return 1000.0 * dv / slope_v_per_s  # ms

def generate_pulse_waveform(p, num_samples=16384):
    v0   = p["baseline"]
    v_up = p["v_up"]
    v_dn = p["v_down"]

    t1 = p["t_flat_before_ms"] / 1000.0
    t2 = p["t_ramp_ms"]        / 1000.0
    t3 = p["t_flat_after_ms"]  / 1000.0

    t_total = t1 + t2 + t3

    t = np.linspace(0, t_total, num_samples)

    y = np.piecewise(
        t,
        [
            t < t1,                     # baseline
            (t >= t1) & (t < t1 + t2),  # ramp down
            t >= t1 + t2                # baseline
        ],
        [
            lambda t: v0,
            lambda t: v_up + (v_dn - v_up) * ((t - t1) / t2),
            lambda t: v0
        ]
    )

    freq = 1 / t_total

    return t, y, freq
def generate_three_point_cycle(p, samples_per_step=2048):
    """
    Build one full cycle:  V1 -> V2 -> V3 -> V1
    Each leg (Vx -> Vy) has the same timing:
        t_pre  (flat at Vx)
        t_rise (ramp Vx -> Vy)
        t_post (flat at Vy)
        t_ret  (extra dwell at Vy before next leg)
    """

    v1 = float(p["v1"])
    v2 = float(p["v2"])
    v3 = float(p["v3"])

    t_pre  = p["t_pre_ms"]  / 1000.0
    t_rise = p["t_rise_ms"] / 1000.0
    t_post = p["t_post_ms"] / 1000.0
    t_ret  = p["t_ret_ms"]  / 1000.0

    seg_time = t_pre + t_rise + t_post + t_ret

    samples_per_seg = samples_per_step
    current_t = 0.0

    def one_segment(v_start, v_end):
        nonlocal current_t

        t_local = np.linspace(0, seg_time, samples_per_seg, endpoint=False)
        t_global = t_local + current_t

        # piecewise: flat (pre), ramp (rise), flat (post+ret)
        waveform = np.piecewise(
            t_local,
            [
                t_local < t_pre,
                (t_local >= t_pre) & (t_local < t_pre + t_rise),
                t_local >= t_pre + t_rise,
            ],
            [
                lambda t: v_start,
                lambda t: v_start + (v_end - v_start) * (t - t_pre) / t_rise if t_rise > 0 else v_end,
                lambda t: v_end,
            ],
        )

        current_t += seg_time
        return t_global, waveform

    # V1 -> V2
    t12, y12 = one_segment(v1, v2)
    # V2 -> V3
    t23, y23 = one_segment(v2, v3)
    # V3 -> V1
    t31, y31 = one_segment(v3, v1)

    t_full = np.concatenate([t12, t23, t31])
    y_full = np.concatenate([y12, y23, y31])

    total_time = t_full[-1] - t_full[0] + (seg_time / samples_per_seg)
    freq = 1.0 / total_time if total_time > 0 else 0.0

    return t_full, y_full, freq

def update_parameters():
    for key in ["awg_vpp", "awg_offset"]:
        try:
            params[key] = float(dpg.get_value(key))
        except:
            pass
# === Sweep Executor with Pause/Resume ===
def generate_sweep():
    global full_waveform, combined_freq

    x_vals.clear()
    segments.clear()
    saved_sweep_steps.clear()
    segment_times.clear()
    segment_bounds.clear()
    x_vals_resampled.clear()
    y_vals_resampled.clear()

    # remove highlight box if it exists
    if dpg.does_item_exist("highlight_box"):
        dpg.delete_item("highlight_box")

    dpg.set_value("sweep_plot_series", [[], []])

    # ----- initial waveform -----
    interp_params = waveform1_params.copy()
    current_time = 0.0

    t, y, freq = generate_pulse_waveform(waveform1_params, samples_per_step)
    seg_duration = t[-1]

    segments.append(y)
    segment_times.append((0.0, seg_duration))
    saved_sweep_steps.append(waveform1_params.copy())
    segment_bounds.append((0.0, seg_duration, float(np.min(y)), float(np.max(y))))
    x_vals.extend(t)

    current_time = seg_duration

    # ----- sweep loop -----
    counter = 0
    while True:
        counter += 1

        # interpolate each sweepable parameter
        for key in sweep_keys:
            v1 = float(waveform1_params[key])
            v2 = float(waveform2_params[key])
            inc = increments.get(key, 0.0)
            step_amt = inc * counter

            # monotonic interpolation with clamping
            if v1 == v2:
                interp_params[key] = v2
            elif v2 > v1:
                interp_params[key] = min(v1 + step_amt, v2)
            else:
                interp_params[key] = max(v1 - step_amt, v2)

        # generate pulse at this step
        t_seg, y_seg, _ = generate_pulse_waveform(interp_params, samples_per_step)
        seg_duration = t_seg[-1]

        # shift time axis
        t_seg = t_seg + current_time

        segments.append(y_seg)
        segment_times.append((current_time, current_time + seg_duration))
        saved_sweep_steps.append(interp_params.copy())
        segment_bounds.append(
            (current_time, current_time + seg_duration,
             float(np.min(y_seg)), float(np.max(y_seg)))
        )

        x_vals.extend(t_seg)
        current_time += seg_duration

        # stop when ALL params have reached their final values
        done = True
        for key in sweep_keys:
            if abs(interp_params[key] - waveform2_params[key]) > 1e-9:
                done = False
                break

        if done:
            break

    full_waveform = np.concatenate(segments)
    total_time = current_time
    combined_freq = 1.0 / total_time if total_time > 0 else 0.0

    # update GUI plot
    dpg.set_value("sweep_plot_series", [np.array(x_vals), full_waveform])

    return np.array(x_vals), full_waveform, combined_freq

def resample_waveform_to_uniform_time(x_vals, y_vals, num_points=16384):
    total_time = x_vals[-1] - x_vals[0]
    new_x = np.linspace(x_vals[0], x_vals[-1], num_points)
    interpolator = interp1d(x_vals, y_vals, kind='linear')
    new_y = interpolator(new_x)
    return new_x, new_y, total_time   

def get_frequency_from_params(p, full_cycle=True):
    total_duration = (p["start_v_ms"] +
                      p["slope_up_ms"] +
                      p["end_v_ms"]) / 1000

    if full_cycle:
        return 1 / (2 * total_duration) if total_duration > 0 else 0
    else:
        return 1 / total_duration if total_duration > 0 else 0


def deploy_waveform():
    if not awg_connected:
        reconnect_awg(dpg.get_value("awg_addr"))
        if not awg_connected:
            dpg.set_value("status_text", "No AWG connected.")
            return
    try:
        t_waveform, waveform, frequency = generate_waveform()
        duration_s = t_waveform[-1]

        # --- Prepare waveform for 33220A ---
        # Your GUI waveform is 0 → 1 V.
        # 33220A expects -1 → +1 in DATA VOLATILE format.
        wave = np.array(waveform, dtype=float)

        # Convert 0–1 range → -1–+1 range
        wave = 2.0 * wave - 1.0
        wave = np.clip(wave, -1.0, 1.0)

        # Create ASCII list (float format OK for 33220A)
        values_str = ",".join(f"{v:.4f}" for v in wave)

        # Clear errors
        awg.write("*CLS")
        # awg.write("FUNC:ARB:INT ON")

        # --- Upload entire waveform in one command ---
        awg.write("DATA VOLATILE," + values_str)

        # Check for upload errors
        err_upload = awg.query("SYST:ERR?").strip()
        print("ERR after upload:", err_upload)

        # --- Select and enable the waveform ---
        awg.write("FUNC:USER VOLATILE")
        awg.write("FUNC USER")

        # Set output to 0–1 V range:
        # VOLT=1 Vpp and OFFS=0.5 V maps -1→0 V and +1→1 V
        awg.write("VOLT 1")
        awg.write("VOLT:OFFS 0.5")

        # --- Scale output to match v_start and v_end ---
        vmin = float(params["v_start"])
        vmax = float(params["v_end"])

        vpp = (vmax) - (vmin)
        offset = (vmax + vmin) / 2.0
    
        if vpp == 0:
            vpp = 0.001  # prevent AWG error if user enters same start/end

        # awg.write(f"VOLT {params['awg_vpp']}")
        awg.write(f"VOLT {vpp}")
        awg.write(f"VOLT:OFFS {params['awg_offset']}")
        # Set frequency from total duration
        awg.write(f"FREQ {frequency}")

        awg.write("OUTP ON")

        dpg.set_value(
            "status_text",
            f"Uploaded {len(wave)} pts | T={duration_s:.4f}s | f≈{frequency:.3f}Hz | ERR={err_upload}"
        )

    except Exception as e:
        dpg.set_value("status_text", f"Error: {e}")

def start_sweep():
    print("im here in start_sweep")
    generate_sweep();
    update_waveform(1);
    update_waveform(2);    
    update_increments()
    # plt.plot(full_waveform)
    # plt.title("Sweep waveform preview")
    # plt.show()
    print("entering deploy sweep")
    try:
        params["output_channel"] = int(dpg.get_value("output_channel"))
    except Exception as e:
        print(f"[ERROR] Failed to read IP or output channel from GUI: {e}")
        dpg.set_value("status_text", "Invalid Red Pitaya IP or output channel.")
        return
    deploy_sweep();

def toggle_pause():
    global sweep_paused
    sweep_paused = not sweep_paused
    dpg.set_value("status_text", "Paused" if sweep_paused else "Resumed")

def stop_sweep():
    global sweep_running, sweep_paused
    sweep_running = False
    sweep_paused = False
    dpg.set_value("status_text", "Sweep stopped manually.")

def deploy_sweep():
    try:
        global full_waveform, combined_freq, x_vals

        waveform = np.array(full_waveform, dtype=float)
        frequency = combined_freq
        duration_s = float(x_vals[-1])

        # Scale to -1..+1
        vmin = waveform.min()
        vmax = waveform.max()
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6

        wave_norm = 2 * (waveform - vmin) / (vmax - vmin) - 1
        wave_norm = np.clip(wave_norm, -1, 1)

        # --- FIX: Downsample to AWG limits (8192 points) ---
        MAX_POINTS = 65536 #65536
        # if len(wave_norm) > MAX_POINTS:
        #     idx = np.linspace(0, len(wave_norm)-1, MAX_POINTS).astype(int)
        #     wave_norm = wave_norm[idx]
        #     print(f"Downsampled waveform to {MAX_POINTS} points.")
        if len(wave_norm) > MAX_POINTS:
            x_old = np.linspace(0, 1, len(wave_norm))
            x_new = np.linspace(0, 1, MAX_POINTS)
            wave_norm = np.interp(x_new, x_old, wave_norm)
            print(f"Resampled waveform to {MAX_POINTS} points.")    

        # Convert to CSV format
        values_str = ",".join(f"{v:.6f}" for v in wave_norm)

        awg.write("*CLS")
        awg.write("FUNC USER")
        awg.write("FUNC:USER VOLATILE")
        awg.write("DATA VOLATILE," + values_str)

        err = awg.query("SYST:ERR?")
        print("AWG message:", err)

        # output amplitude
        awg_vpp = vmax - vmin
        awg_offset = (vmax + vmin) / 2
        if awg_vpp < 1e-3:
            awg_vpp = 1e-3

        awg.write(f"VOLT {awg_vpp}")
        awg.write(f"VOLT:OFFS {awg_offset}")

        awg.write(f"FREQ {frequency}")

        ch = int(params["output_channel"])
        awg.write(f"OUTP{ch} ON")

        dpg.set_value("status_text",
                      f"Sweep uploaded ({len(wave_norm)} pts) | Duration={duration_s:.3f}s | f={frequency:.3f} Hz")

    except Exception as e:
        dpg.set_value("status_text", f"AWG ERROR: {str(e)}")
        print("AWG deploy error:", e)

# === UI Sync & Handlers ===
def update_waveform(tab_id):
    global waveform1_params, waveform2_params

    param_set = waveform1_params if tab_id == 1 else waveform2_params
    param_set["v_down"] = param_set["baseline"]


    # pull GUI values back into param_set
    for key in param_set:
        if key in ["awg_vpp", "awg_offset", "output_channel"]:
            continue
        widget_id = f"{key}_{tab_id}"
        if dpg.does_item_exist(widget_id):
            try:
                param_set[key] = float(dpg.get_value(widget_id))
            except:
                pass

    # ----------- NEW: Auto-recompute ramp time -----------
    # slope is defined ONLY by tab 1 (initial waveform)
    v_low_1 = min(waveform1_params["v_up"], waveform1_params["v_down"], waveform1_params["baseline"])
    v_high_1 = max(waveform1_params["v_up"], waveform1_params["v_down"], waveform1_params["baseline"])
    freq_1 = waveform1_params["initial_freq_hz"]

    slope_ref = triangle_slope_from_freq(v_low_1, v_high_1, freq_1)

    new_t_ramp = compute_ramp_time_from_slope(
        param_set["v_up"],
        param_set["v_down"],
        slope_ref
    )
    param_set["t_ramp_ms"] = new_t_ramp

    # reflect updated t_ramp_ms back into GUI (read-only field)
    dpg.set_value(f"t_ramp_ms_{tab_id}", new_t_ramp)
    # ------------------------------------------------------

    t, y, _ = generate_pulse_waveform(param_set, samples_per_step)
    series = "plot1_series" if tab_id == 1 else "plot2_series"
    dpg.set_value(series, [t, y])

def update_increments():
    global increments

    for key in increments.keys():
        widget_id = f"inc_{key}"
        try:
            val = dpg.get_value(widget_id)
            if val is not None:
                increments[key] = float(val)
        except Exception as e:
            print(f"[WARNING] Could not update increment for '{key}' from widget '{widget_id}': {e}")

def reconnect_awg(address):
    global awg
    try:
        awg = rm.open_resource(address)
        # awg.timeout = 20000 # added this, 20 sec
        dpg.set_value("status_text", "AWG reconnected: " + awg.query("*IDN?"))
    except Exception as e:
        dpg.set_value("status_text", f"Failed to connect: {e}")

def add_inputs(tab_id, values):
    
    with dpg.table(header_row=False, borders_innerV=True, borders_innerH=True):
        dpg.add_table_column()
        dpg.add_table_column()

        with dpg.table_row():
            
            with dpg.group():
                dpg.add_text("Voltage Values")

                for key in ["v_up", "baseline"]:
                # for key in ["v_up", "v_down", "baseline"]:

                    dpg.add_text(f"{key}:")
                    dpg.add_input_float(
                        default_value=values[key],
                        tag=f"{key}_{tab_id}",
                        width=120,
                        callback=lambda s,a,u=key: update_waveform(tab_id),
                        format="%.4f",
                    )

                # Tab 1 only: initial frequency
                if tab_id == 1:
                    dpg.add_text("Initial Frequency (Hz):")
                    dpg.add_input_float(
                        default_value=values["initial_freq_hz"],
                        tag=f"initial_freq_hz_{tab_id}",
                        width=120,
                        callback=lambda s,a: update_waveform(1),
                        format="%.4f",
                    )
            with dpg.group():
                dpg.add_text("Timing (ms)")

                # # before-flat
                # dpg.add_text("t_flat_before_ms:")
                # dpg.add_input_float(
                #     default_value=values["t_flat_before_ms"],
                #     tag=f"t_flat_before_ms_{tab_id}",
                #     width=120,
                #     callback=lambda s,a: update_waveform(tab_id),
                #     format="%.4f",
                # )

                # # after-flat
                # dpg.add_text("t_flat_after_ms:")
                # dpg.add_input_float(
                #     default_value=values["t_flat_after_ms"],
                #     tag=f"t_flat_after_ms_{tab_id}",
                #     width=120,
                #     callback=lambda s,a: update_waveform(tab_id),
                #     format="%.4f",
                # )

                #  THIS ONE is REQUIRED or update_waveform will always crash
                dpg.add_text("t_ramp_ms:")
                dpg.add_input_float(
                    default_value=values["t_ramp_ms"],
                    tag=f"t_ramp_ms_{tab_id}",
                    width=120,
                    readonly=True,
                    enabled=False,
                    format="%.4f",
                )

            """with dpg.group():
                dpg.add_text("Timing (ms)")

                for key in ["t_flat_before_ms", "t_flat_after_ms"]:
                    dpg.add_text(f"{key}:")
                    dpg.add_input_float(
                        default_value=values[key],
                        tag=f"{key}_{tab_id}",
                        width=120,
                        callback=lambda s,a,u=key: update_waveform(tab_id),
                        format="%.4f",
                    )

                # t_ramp_ms — read-only
                dpg.add_text("t_ramp_ms:")
                dpg.add_input_float(
                    default_value=values["t_ramp_ms"],
                    tag=f"t_ramp_ms_{tab_id}",
                    width=120,
                    readonly=True,            # makes it dulled out
                    enabled=False,            # disables editing
                    format="%.4f",
                )
            """

def add_increment_inputs():
    def make_step_button(key, delta):
        return lambda: (
            dpg.set_value(f"inc_{key}", round(float(dpg.get_value(f"inc_{key}") or 0) + delta, 6)),
            update_increments()
        )

    with dpg.table(header_row=False, borders_innerV=True, borders_innerH=True):
        dpg.add_table_column(init_width_or_weight=20)   # Label
        dpg.add_table_column(init_width_or_weight=15)   # Input
        dpg.add_table_column(init_width_or_weight=5)   # Minus
        dpg.add_table_column(init_width_or_weight=80)   # Plus

        for key in increments.keys():
            with dpg.table_row():
                dpg.add_text(f"{key}:", wrap=0)
                dpg.add_input_float(
                    tag=f"inc_{key}",
                    default_value=increments[key],
                    width=60,
                    step=0.0,  # disables built-in +/- buttons
                    callback=update_increments,
                    format="%.3f"
                )
                dpg.add_button(label="-", width=20, height=20, callback=make_step_button(key, -0.01))
                dpg.add_button(label="+", width=20, height=20, callback=make_step_button(key, +0.01))

def on_sweep_select(sender, app_data):
    x1, x2 = app_data  # x range of selected region
    print(f"Selected x range: {x1:.3f} to {x2:.3f}")

    # Search through the saved segments
    current_time = 0
    for i, segment in enumerate(saved_sweep_steps):  # make sure you saved them earlier
        _, _, seg_duration = generate_waveform(segment)
        if current_time <= x1 <= x2 <= current_time + seg_duration:
            # Match found
            lines = [f"{k}: {v:.3f}" for k, v in segment.items() if k not in ["output_channel"]]
            dpg.set_value("waveform_output_text", "\n".join(lines))
            
        current_time += seg_duration

def hover_plot_tracker():
    if dpg.is_item_hovered("sweep_plot_series"):
        mouse_pos = dpg.get_mouse_pos(local=False)
        plot_x, _ = dpg.get_plot_mouse_pos()

        # Find which segment it falls into
        for i, (start, end) in enumerate(segment_times):
            if start <= plot_x <= end:
                seg = saved_sweep_steps[i]
                lines = [f"{k}: {v:.3f}" for k, v in seg.items() if k not in ["output_channel"]]
                dpg.set_value("waveform_output_text", "\n".join(lines))
                return

    # If not hovering or outside bounds
    dpg.set_value("waveform_output_text", "")

def update_hover_info():
    plot_x, _ = dpg.get_plot_mouse_pos()
    for i, (start, end) in enumerate(segment_times):
        if start <= plot_x <= end:
            seg = saved_sweep_steps[i]
            lines = [f"{k}: {v:.3f}" for k, v in seg.items() if k not in ["rp_ip", "output_channel"]]
            dpg.set_value("waveform_output_text", "\n".join(lines))
            return
    dpg.set_value("waveform_output_text", "")  # Clear if not hovering

def on_left_click():
    if dpg.is_item_hovered("sweep_plot"):
        plot_x, plot_y = dpg.get_plot_mouse_pos()
        for i, (x_start, x_end, y_min, y_max) in enumerate(segment_bounds):
            if x_start <= plot_x <= x_end and y_min - 0.05 <= plot_y <= y_max + 0.05:  # buffer
                seg = saved_sweep_steps[i]
                lines = [f"{k}: {v:.3f}" for k, v in seg.items() if k not in ["rp_ip", "output_channel"]]
                dpg.set_value("waveform_output_text", "\n".join(lines))
                highlight_segment(i, x_start, x_end, y_min, y_max)
                return

        dpg.set_value("waveform_output_text", "Clicked outside waveform.")

def on_plot_clicked(sender, app_data, user_data):
    if dpg.is_item_clicked("sweep_plot_series", button=dpg.mvMouseButton_Left):
        plot_x, _ = dpg.get_plot_mouse_pos()
        for i, (start, end) in enumerate(segment_times):
            if start <= plot_x <= end:
                seg = saved_sweep_steps[i]
                lines = [f"{k}: {v:.3f}" for k, v in seg.items() if k not in ["rp_ip", "output_channel"]]
                dpg.set_value("waveform_output_text", "\n".join(lines))
                return
        dpg.set_value("waveform_output_text", "Clicked outside segment range.")

def highlight_segment(index, x_start, x_end, y_min, y_max):
    # Delete any existing highlight box
    if dpg.does_item_exist("highlight_box"):
        dpg.delete_item("highlight_box")

    margin = 0.02  # Small vertical margin for clarity

    # Create a new draw layer with just a bordered box (no fill)
    with dpg.draw_layer(parent="sweep_plot", tag="highlight_box"):
        dpg.draw_rectangle(
            pmin=(x_start/ y_min - margin),
            pmax=(x_end, y_max + margin),
            color=(128, 0, 255, 180),  # Light purple border
            thickness=0.005
        )

# === GUI Builder ===
def build_gui():
    dpg.create_context()
    with dpg.window(label="Full Sweep GUI", width=1000, height=850):
        dpg.add_text("Waveform Sweep for Dark Pulse Soliton", color=(0, 200, 255), wrap=500)
        dpg.add_input_text(label="Red Pitaya IP", default_value=params["rp_ip"], tag="rp_ip")
        dpg.add_input_text(
            label="AWG VISA Address",
            default_value=AWG_RESOURCE,
            tag="awg_addr",
            callback=lambda: reconnect_awg(dpg.get_value("awg_addr")),
        )
        dpg.add_button(label="Reconnect AWG", callback=lambda:
                       reconnect_awg(dpg.get_value("awg_addr")))
        dpg.add_combo(label="Output Channel", items=["1", "2"], default_value="1", tag="output_channel")
        # dpg.add_button(label="Start Sweep", callback=lambda: threading.Thread(target=start_sweep).start())
        dpg.add_button(label="Start Sweep", callback=start_sweep)
        # dpg.add_button(label="Pause / Resume", callback=toggle_pause)
        # dpg.add_button(label="Stop Sweep", callback=stop_sweep)
        dpg.add_text("", tag="status_text")

        with dpg.table(header_row=True):
            dpg.add_table_column(label="Parameter")
            dpg.add_table_column(label="Value")

            for key in ["awg_vpp", "awg_offset"]:
                with dpg.table_row():
                    dpg.add_text(key.replace("_", " ").title())
                    dpg.add_input_float(
                        tag=key,
                        default_value=params[key],
                        callback=update_parameters
                    )
        dpg.add_separator()
        dpg.add_text("Sweep Increments:")
        add_increment_inputs()
        with dpg.group():
            dpg.add_text("Global Timing (ms)")

            dpg.add_text("t_flat_before_ms:")
            dpg.add_input_float(
                default_value=waveform1_params["t_flat_before_ms"],
                tag="t_flat_before_ms_global",
                width=120,
                callback=lambda s,a: update_global_flat_times(),
                format="%.4f",
            )

            dpg.add_text("t_flat_after_ms:")
            dpg.add_input_float(
                default_value=waveform1_params["t_flat_after_ms"],
                tag="t_flat_after_ms_global",
                width=120,
                callback=lambda s,a: update_global_flat_times(),
                format="%.4f",
            )

        with dpg.tab_bar():
            with dpg.tab(label="Initial Waveform"):
                add_inputs(1, waveform1_params)
                with dpg.plot(label="Initial", height=250, width=900):
                    dpg.add_plot_axis(dpg.mvXAxis)
                    dpg.add_plot_axis(dpg.mvYAxis, tag="axis1")
                    dpg.add_line_series([], [], parent="axis1", tag="plot1_series")
                    dpg.set_axis_limits("axis1", y_min, y_max)

            with dpg.tab(label="Final Waveform"):
                add_inputs(2, waveform2_params)
                with dpg.plot(label="Final", height=250, width=900):
                    dpg.add_plot_axis(dpg.mvXAxis)
                    dpg.add_plot_axis(dpg.mvYAxis, tag="axis2")
                    dpg.add_line_series([], [], parent="axis2", tag="plot2_series")
                    dpg.set_axis_limits("axis2", y_min, y_max)

            with dpg.tab(label="Sweep Visual"):
                with dpg.group():
                    with dpg.plot(label="Sweep Preview", height=300, width=900, tag="sweep_plot"):
                        dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_sweep")
                        dpg.add_plot_axis(dpg.mvYAxis, tag="axis_sweep")
                        dpg.add_line_series([], [], parent="axis_sweep", tag="sweep_plot_series")
                        dpg.set_axis_limits("axis_sweep", y_min, y_max)
                        dpg.draw_layer(parent="sweep_plot", tag="highlight_box")  # For drawing the highlight

                # Add text display for clicked info
                dpg.add_text("Clicked Waveform Info:", tag="waveform_output_header")
                dpg.add_text("", tag="waveform_output_text")

                # Add the click handler
                with dpg.handler_registry():
                    dpg.add_mouse_click_handler(callback=lambda: on_left_click())

    dpg.create_viewport(title='Waveform Sweep GUI (Pause/Resume/Live)', width=1000, height=850)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    update_waveform(1)
    update_waveform(2)
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    build_gui()

