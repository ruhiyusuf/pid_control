import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import dearpygui.dearpygui as dpg
from redpitaya_scpi import scpi
from scipy.interpolate import interp1d

# === Global Parameters ===
params = {
    "v_start": 0.0, "v_end": 1.0, "v_down_end": 1.0,
    "start_v_ms": 20.0, "slope_up_ms": 20.0, "slope_down_ms": 40.0, "end_v_ms": 500.0,
    "rp_ip": "rp-f0cbc6.local", "output_channel": 1
}

waveform1_params = params.copy()
waveform2_params = params.copy()
increments = {
    "v_start": 0.05, "v_end": 0.05, "v_down_end": 0.05,
    "start_v_ms": 10.0, "slope_up_ms": 10.0, "slope_down_ms": 10.0, "end_v_ms": 50.0
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

# === Waveform Generator ===
def generate_waveform(p, num_samples=16384):
    s = p["start_v_ms"] / 1000
    u = p["slope_up_ms"] / 1000
    d = p["slope_down_ms"] / 1000
    e = p["end_v_ms"] / 1000
    total_time = s + u + d + e
    t = np.linspace(0, total_time, num_samples)

    waveform = np.piecewise(t,
        [t < s,
         (t >= s) & (t < s + u),
         (t >= s + u) & (t < s + u + d),
         t >= s + u + d],
        [lambda t: p["v_start"],
         lambda t: p["v_start"] + (p["v_end"] - p["v_start"]) * (t - s) / u,
         lambda t: p["v_end"] + (p["v_down_end"] - p["v_end"]) * (t - s - u) / d,
         lambda t: p["v_down_end"]])
    return t, waveform, total_time

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

    # Delete previous highlight box
    if dpg.does_item_exist("highlight_box"):
        dpg.delete_item("highlight_box")

    # Clear old waveform plot
    dpg.set_value("sweep_plot_series", [[], []])

    interp_params = waveform1_params.copy()

    current_time = 0

    # === Generate and save the initial waveform exactly as-is ===
    t, segment, segment_time = generate_waveform(waveform1_params,
                                                 num_samples=samples_per_step)
    segments.append(segment)
    segment_times.append((0.0, segment_time))
    saved_sweep_steps.append(waveform1_params.copy())
    segment_bounds.append((0.0, segment_time, min(segment), max(segment)))
    x_vals.extend(np.linspace(0.0, segment_time, len(segment)))

    current_time = segment_time  # Start the sweep from here

    # === Start the actual incremental sweep from here ===
    counter = 0

    while True:
        counter += 1
        for key in waveform1_params:
            if key in ["rp_ip", "output_channel"]:
                continue

            val1 = float(waveform1_params[key])
            val2 = float(waveform2_params[key])
            amt = counter * increments[key]

            if val1 != val2:
                if val2 > val1:
                    interp_val = val2 if (val1 + amt > val2) else val1 + amt
                else:
                    interp_val = val2 if (val1 - amt < val2) else val1 - amt
            else:
                interp_val = val2

            interp_params[key] = interp_val

        _, segment, segment_time = generate_waveform(interp_params,
                                                     num_samples=samples_per_step)
        start_time = current_time
        end_time = current_time + segment_time

        segments.append(segment)
        segment_times.append((start_time, end_time))
        saved_sweep_steps.append(interp_params.copy())

        y_min_seg = min(segment)
        y_max_seg = max(segment)
        segment_bounds.append((start_time, end_time, y_min_seg, y_max_seg))

        # Generate actual time values for this segment
        t_segment = np.linspace(start_time, end_time, len(segment))
        x_vals.extend(t_segment)
        current_time = end_time 

        if waveform2_params == interp_params:
            break

    # Combine waveform segments and compute total time
    full_waveform = np.concatenate(segments)
    total_time = current_time
    combined_freq = 1 / total_time

    # Update GUI plot
    dpg.set_value("sweep_plot_series", [np.array(x_vals), full_waveform])
    print("full_waveform ", full_waveform)
    print("full waveform length ", len(full_waveform))
    print("x vals ", np.array(x_vals))
    print("x vals length ", len(x_vals))

def resample_waveform_to_uniform_time(x_vals, y_vals, num_points=16384):
    total_time = x_vals[-1] - x_vals[0]
    new_x = np.linspace(x_vals[0], x_vals[-1], num_points)
    interpolator = interp1d(x_vals, y_vals, kind='linear')
    new_y = interpolator(new_x)
    return new_x, new_y, total_time   

def get_frequency_from_params(p, full_cycle=True):
    total_duration = (p["start_v_ms"] +
                      p["slope_up_ms"] +
                      p["slope_down_ms"] +
                      p["end_v_ms"]) / 1000  # ms → s

    if full_cycle:
        return 1 / (2 * total_duration) if total_duration > 0 else 0
    else:
        return 1 / total_duration if total_duration > 0 else 0

def deploy_sweep():
    try:
        rp = scpi(params["rp_ip"])
        ch = int(params["output_channel"])
        assert ch in [1, 2], "Output channel must be 1 or 2"
        # rp.tx_txt("GEN:RST")
        rp.tx_txt(f"SOUR{ch}:FUNC ARBITRARY")

        x_vals_resampled, y_vals_resampled, total_time = resample_waveform_to_uniform_time(np.array(x_vals), full_waveform)
        rp.tx_txt(f"SOUR{ch}:TRAC:DATA:DATA " + ",".join(map(str, y_vals_resampled)))
        # plt.plot(x_vals, full_waveform)
        # plt.title("Sweep waveform preview")
        # plt.show()

        rp.tx_txt(f"SOUR{ch}:FREQ:FIX {combined_freq}")
        rp.tx_txt(f"SOUR{ch}:VOLT:OFFS 0")
        rp.tx_txt(f"OUTPUT{ch}:STATE ON")
        rp.tx_txt(f"SOUR{ch}:TRIG:INT")
        rp.close()
        dpg.set_value("status_text", f"Waveform deployed to Red Pitaya (Channel {ch})")
    except OSError as e:
        dpg.set_value("status_text", "Error: Red Pitaya not reachable. Make sure it is properly connected, and wait 20-30s and retry.")
        print("OSError:", e)

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

# === UI Sync & Handlers ===
def update_waveform(tab_id):
    global waveform1_params, waveform2_params

    param_set = waveform1_params if tab_id == 1 else waveform2_params
    for key in param_set:
        if key in ["rp_ip", "output_channel"]:
            continue
        try:
            param_set[key] = dpg.get_value(f"{key}_{tab_id}")
        except Exception as e:
            print(f"[WARNING] Could not update key '{key}': {e}")

    t, y, _ = generate_waveform(param_set)

    if tab_id == 1:
        dpg.set_value("plot1_series", [t, y])
    else:
        dpg.set_value("plot2_series", [t, y])

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

def add_inputs(tab_id, values):
    with dpg.table(header_row=False, borders_innerV=True, borders_innerH=True):
        # Outer table has two columns: voltage and timing
        dpg.add_table_column(init_width_or_weight=100)
        dpg.add_table_column(init_width_or_weight=100)

        with dpg.table_row():
            # === Voltage Table ===
            with dpg.group():
                with dpg.table(header_row=False, borders_innerV=True):
                    dpg.add_table_column(init_width_or_weight=60)
                    dpg.add_table_column(init_width_or_weight=60)
                    with dpg.table_row():
                        dpg.add_text("Voltage Values")
                    for key in ["v_start", "v_end", "v_down_end"]:
                        with dpg.table_row():
                            dpg.add_text(f"{key}:")
                            dpg.add_input_float(
                                default_value=values[key],
                                tag=f"{key}_{tab_id}",
                                width=100,
                                callback=lambda s, a, u=key: update_waveform(tab_id),
                                format="%.3f"
                            )

            # === Timing Table ===
            with dpg.group():
                with dpg.table(header_row=False, borders_innerV=True):
                    dpg.add_table_column(init_width_or_weight=30)
                    dpg.add_table_column(init_width_or_weight=30)
                    with dpg.table_row():
                        dpg.add_text("Timing Parameters")
                    for key in ["start_v_ms", "slope_up_ms", "slope_down_ms", "end_v_ms"]:
                        with dpg.table_row():
                            dpg.add_text(f"{key}:")
                            dpg.add_input_float(
                                default_value=values[key],
                                tag=f"{key}_{tab_id}",
                                width=100,
                                callback=lambda s, a, u=key: update_waveform(tab_id),
                                format="%.3f"
                            )

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
            lines = [f"{k}: {v:.3f}" for k, v in segment.items() if k not in ["rp_ip", "output_channel"]]
            dpg.set_value("waveform_output_text", "\n".join(lines))
            break
        current_time += seg_duration

def hover_plot_tracker():
    if dpg.is_item_hovered("sweep_plot_series"):
        mouse_pos = dpg.get_mouse_pos(local=False)
        plot_x, _ = dpg.get_plot_mouse_pos()

        # Find which segment it falls into
        for i, (start, end) in enumerate(segment_times):
            if start <= plot_x <= end:
                seg = saved_sweep_steps[i]
                lines = [f"{k}: {v:.3f}" for k, v in seg.items() if k not in ["rp_ip", "output_channel"]]
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
            pmin=(x_start, y_min - margin),
            pmax=(x_end, y_max + margin),
            color=(128, 0, 255, 180),  # Light purple border
            thickness=0.005
        )

# === GUI Builder ===
def build_gui():
    dpg.create_context()
    with dpg.window(label="Full Sweep GUI", width=1000, height=850):
        dpg.add_input_text(label="Red Pitaya IP", default_value=params["rp_ip"], tag="rp_ip")
        dpg.add_combo(label="Output Channel", items=["1", "2"], default_value="1", tag="output_channel")
        # dpg.add_button(label="Start Sweep", callback=lambda: threading.Thread(target=start_sweep).start())
        dpg.add_button(label="Start Sweep", callback=start_sweep)
        # dpg.add_button(label="Pause / Resume", callback=toggle_pause)
        # dpg.add_button(label="Stop Sweep", callback=stop_sweep)
        dpg.add_text("", tag="status_text")
        dpg.add_separator()
        dpg.add_text("Sweep Increments:")
        add_increment_inputs()

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

