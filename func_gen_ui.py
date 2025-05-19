import numpy as np
import dearpygui.dearpygui as dpg
from redpitaya_scpi import scpi
import time
import json
import os

# Default input parameters
params = {
    "v_start": 0.0,
    "v_end": 1.0,
    "v_down_end": 0.8,
    "start_v_ms": 20.0,
    "slope_up_ms": 20.0,
    "slope_down_ms": 40.0,
    "end_v_ms": 5000.0,
    "rp_ip": "rp-f0cbc6.local",
    "output_channel" : 1
}

# Default increments for each parameter
increments = {key: 0.1 for key in params if key != "rp_ip"}

# Axis limits for zooming
y_min, y_max = -0.2, 1.2  # Default y-axis limits

hold_value = False

SAVE_PATH = "waveform_settings.json"

def generate_waveform():
    start_v_seconds = params["start_v_ms"] / 1000
    slope_up_seconds = params["slope_up_ms"] / 1000
    slope_down_seconds = params["slope_down_ms"] / 1000
    end_v_seconds = params["end_v_ms"] / 1000

    num_samples = 16384
    total_time = start_v_seconds + slope_up_seconds + slope_down_seconds + end_v_seconds
    frequency = 1 / total_time
    t_waveform = np.linspace(0, total_time, num_samples)

    if hold_value:
        waveform = np.full_like(t_waveform, params["v_down_end"])
    else:
        waveform = np.piecewise(t_waveform, 
            [t_waveform < start_v_seconds, 
             (t_waveform >= start_v_seconds) & (t_waveform < start_v_seconds + slope_up_seconds), 
             (t_waveform >= start_v_seconds + slope_up_seconds) & (t_waveform < start_v_seconds + slope_up_seconds + slope_down_seconds),
             t_waveform >= start_v_seconds + slope_up_seconds + slope_down_seconds], 
            [lambda t: params["v_start"],
             lambda t: params["v_start"] + (params["v_end"] - params["v_start"]) * ((t - start_v_seconds) / slope_up_seconds) if slope_up_seconds > 0 else params["v_end"],
             lambda t: params["v_end"] + (params["v_down_end"] - params["v_end"]) * ((t - start_v_seconds - slope_up_seconds) / slope_down_seconds) if slope_down_seconds > 0 else params["v_down_end"],
             lambda t: params["v_down_end"]])

    return t_waveform.tolist(), waveform.tolist(), frequency

def toggle_hold_value_callback():
    global hold_value
    hold_value = dpg.get_value("hold_checkbox")  # Read the actual checkbox state
    deploy_waveform()
    status = "Holding V_Down_End value (DC flat)" if hold_value else "Waveform restored"
    dpg.set_value("status_text", f"Toggle: {status}")

def plot_waveform():
    t_waveform, waveform, _ = generate_waveform()
    dpg.set_value("waveform_plot", [t_waveform, waveform])
    dpg.set_axis_limits_auto("x_axis")  # Keep x-axis auto-scaling
    dpg.set_axis_limits("y_axis", y_min, y_max)  # Use manually set y-limits

def deploy_waveform():
    try:
        t_waveform, waveform, frequency = generate_waveform()
        rp = scpi(params["rp_ip"])
        ch = int(params["output_channel"])
        assert ch in [1, 2], "Output channel must be 1 or 2"
        # rp.tx_txt("GEN:RST")
        rp.tx_txt(f"SOUR{ch}:FUNC ARBITRARY")
        rp.tx_txt(f"SOUR{ch}:TRAC:DATA:DATA " + ",".join(map(str, waveform)))
        rp.tx_txt(f"SOUR{ch}:FREQ:FIX {frequency}")
        rp.tx_txt(f"SOUR{ch}:VOLT:OFFS 0")
        rp.tx_txt(f"OUTPUT{ch}:STATE ON")
        rp.tx_txt(f"SOUR{ch}:TRIG:INT")
        rp.close()
        dpg.set_value("status_text", f"Waveform deployed to Red Pitaya (Channel {ch})")
    except OSError as e:
        dpg.set_value("status_text", "Error: Red Pitaya not reachable. Make sure it is properly connected, and wait 20-30s and retry.")
        print("OSError:", e)

def zoom_in():
    """ Zoom in by decreasing the range of the y-axis """
    global y_min, y_max
    center = (y_max + y_min) / 2
    zoom_factor = 0.8  # 20% zoom in
    new_range = (y_max - y_min) * zoom_factor / 2
    y_min, y_max = center - new_range, center + new_range
    dpg.set_axis_limits("y_axis", y_min, y_max)

def zoom_out():
    """ Zoom out by increasing the range of the y-axis """
    global y_min, y_max
    center = (y_max + y_min) / 2
    zoom_factor = 1.25  # 25% zoom out
    new_range = (y_max - y_min) * zoom_factor / 2
    y_min, y_max = center - new_range, center + new_range

def update_parameters():
    for key in params:
        if key == "rp_ip":
            params[key] = dpg.get_value(key)
        elif key == "output_channel":
            params[key] = int(dpg.get_value(key))  # Still allow int here
        else:
            value = dpg.get_value(key)
            if key in ["start_v_ms", "slope_up_ms", "slope_down_ms", "end_v_ms"]:
                # Clamp to 0 minimum
                value = max(0.0, float(value)) if value is not None else 0.0
                dpg.set_value(key, value)
            params[key] = float(value)
    plot_waveform()
    deploy_waveform()

def update_increments():
    for key in increments:
        raw_val = dpg.get_value(f"inc_{key}")
        try:
            if raw_val is not None and raw_val != "":
                increments[key] = float(raw_val)
                dpg.configure_item(key, step=increments[key])
        except ValueError:
            pass  # User is still typing or entered invalid text temporarily

def save_settings():
    filename = dpg.get_value("filename_input") or "waveform_settings.json"
    if not filename.endswith(".json"):
        filename += ".json"
    try:
        with open(filename, "w") as f:
            json.dump(params, f, indent=4)
        dpg.set_value("status_text", f"Settings saved to '{filename}'")
    except Exception as e:
        dpg.set_value("status_text", f"Error saving: {e}")

def load_settings():
    global params
    filename = dpg.get_value("filename_input") or "waveform_settings.json"
    if not filename.endswith(".json"):
        filename += ".json"
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                params.update(json.load(f))
            for key in params:
                if dpg.does_item_exist(key):
                    dpg.set_value(key, params[key])
            plot_waveform()
            deploy_waveform()
            dpg.set_value("status_text", f"Settings loaded from '{filename}'")
        except Exception as e:
            dpg.set_value("status_text", f"Error loading: {e}")
    else:
        dpg.set_value("status_text", f"File '{filename}' not found.")

dpg.create_context()
dpg.set_global_font_scale(1.2)

with dpg.window(label="Waveform Generator", width=1000, height=800):    
    # load and save buttons
    dpg.add_input_text(label="Save/Load Filename", tag="filename_input", default_value="waveform_settings.json", width=250)

    with dpg.group(horizontal=True):
        dpg.add_button(label="Save Settings", callback=save_settings)
        dpg.add_button(label="Load Settings", callback=load_settings)

    with dpg.collapsing_header(label="Voltage Values"):
        with dpg.table(header_row=True):
            dpg.add_table_column(label="Parameter")
            dpg.add_table_column(label="Value")
            dpg.add_table_column(label="Increment")
            for key in ["v_start", "v_end", "v_down_end"]:
                with dpg.table_row():
                    dpg.add_text(key.replace("_", " ").title())
                    dpg.add_input_float(tag=key, default_value=params[key], step=increments[key], callback=update_parameters)
                    dpg.add_input_float(tag=f"inc_{key}", default_value=increments[key], callback=update_increments, on_enter=True)
    with dpg.collapsing_header(label="Duration"):
        with dpg.table(header_row=True):
            dpg.add_table_column(label="Parameter")
            dpg.add_table_column(label="Value")
            dpg.add_table_column(label="Increment")
            for key in ["start_v_ms", "slope_up_ms", "slope_down_ms", "end_v_ms"]:
                with dpg.table_row():
                    dpg.add_text(key.replace("_", " ").title())
                    dpg.add_input_float(tag=key, default_value=params[key], step=increments[key], callback=update_parameters)
                    dpg.add_input_float(tag=f"inc_{key}", default_value=increments[key], callback=update_increments)
    dpg.add_input_text(label="Red Pitaya IP", tag="rp_ip", default_value=params["rp_ip"], callback=update_parameters)
    dpg.add_text("", tag="status_text")

    with dpg.plot(label="Waveform Plot", height=300, width=500):
        dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="x_axis")
        dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude", tag="y_axis")
        dpg.add_line_series([], [], parent="y_axis", tag="waveform_plot")
        dpg.add_plot_legend()
        dpg.set_axis_limits_auto("x_axis")
        dpg.set_axis_limits("y_axis", y_min, y_max)

    # select output channel
    dpg.add_combo(
        label="Output Channel",
        items=["1", "2"],
        default_value="1",
        tag="output_channel",
        width=80,
        callback=lambda: (
            params.update({"output_channel": int(dpg.get_value("output_channel"))}),
            deploy_waveform()
        )
    )

    # Add toggle button
    dpg.add_checkbox(label="Hold V_Down_End Value", callback=lambda: toggle_hold_value_callback(), tag="hold_checkbox")

    # Add Zoom In and Zoom Out buttons
    dpg.add_button(label="Zoom In", callback=lambda: (
        dpg.set_axis_limits("x_axis", *[lim * 0.8 for lim in dpg.get_axis_limits("x_axis")]),
        dpg.set_axis_limits("y_axis", *[lim * 0.8 for lim in dpg.get_axis_limits("y_axis")])
    ))
    dpg.add_button(label="Zoom Out", callback=lambda: (
        dpg.set_axis_limits("x_axis", *[lim * 1.2 for lim in dpg.get_axis_limits("x_axis")]),
        dpg.set_axis_limits("y_axis", *[lim * 1.2 for lim in dpg.get_axis_limits("y_axis")])
    ))  

dpg.create_viewport(title='Red Pitaya Waveform Editor', width=1000, height=800)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

