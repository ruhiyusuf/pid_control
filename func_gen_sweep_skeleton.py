import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import dearpygui.dearpygui as dpg
from redpitaya_scpi import scpi

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
    segments = []
    segment_durations = []
    x_vals = []
    current_time = 0

    interp_params = waveform1_params.copy()
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

        _, segment, segment_time = generate_waveform(interp_params, num_samples=samples_per_step)
        segments.append(segment)
        segment_durations.append(segment_time)

        # Generate actual time values for this segment
        t_segment = np.linspace(current_time, current_time + segment_time, len(segment))
        x_vals.extend(t_segment)
        current_time += segment_time

        if waveform2_params == interp_params:
            break

    # Combine waveform segments and compute total time
    full_waveform = np.concatenate(segments)
    total_time = current_time
    combined_freq = 1 / total_time

    # Update GUI plot
    dpg.set_value("sweep_plot_series", [np.array(x_vals), full_waveform])
    
def deploy_sweep():
    try:
        rp = scpi(params["rp_ip"])
        ch = int(params["output_channel"])
        assert ch in [1, 2], "Output channel must be 1 or 2"
        # rp.tx_txt("GEN:RST")
        rp.tx_txt(f"SOUR{ch}:FUNC ARBITRARY")
        rp.tx_txt(f"SOUR{ch}:TRAC:DATA:DATA " + ",".join(map(str, full_waveform)))
        # plt.plot(full_waveform)
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
    # update_waveform(1);
    # update_waveform(2);    
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
                with dpg.plot(label="Sweep Preview", height=300, width=900):
                    dpg.add_plot_axis(dpg.mvXAxis)
                    dpg.add_plot_axis(dpg.mvYAxis, tag="axis_sweep")
                    dpg.add_line_series([], [], parent="axis_sweep", tag="sweep_plot_series")
                    dpg.set_axis_limits("axis_sweep", y_min, y_max)

    dpg.create_viewport(title='Waveform Sweep GUI (Pause/Resume/Live)', width=1000, height=850)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    update_waveform(1)
    update_waveform(2)
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    build_gui()

