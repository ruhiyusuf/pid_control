import numpy as np
import dearpygui.dearpygui as dpg
from redpitaya_scpi import scpi
import time

# Default input parameters
params = {
    "v_start": 0.0,
    "v_end": 1.0,
    "v_down_end": 1.0,
    "start_v_ms": 20.0,
    "slope_up_ms": 6.0,
    "slope_down_ms": 0.0,
    "end_v_ms": 5000.0,
    "rp_ip": "rp-f0cbc6.local"
}

# Default increments for each parameter
increments = {key: 0.1 for key in params if key != "rp_ip"}

def generate_waveform():
    start_v_seconds = params["start_v_ms"] / 1000
    slope_up_seconds = params["slope_up_ms"] / 1000
    slope_down_seconds = params["slope_down_ms"] / 1000
    end_v_seconds = params["end_v_ms"] / 1000

    num_samples = 16384
    total_time = start_v_seconds + slope_up_seconds + slope_down_seconds + end_v_seconds
    frequency = 1 / total_time
    t_waveform = np.linspace(0, total_time, num_samples)

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

def plot_waveform():
    t_waveform, waveform, _ = generate_waveform()
    dpg.set_value("waveform_plot", [t_waveform, waveform])
    dpg.set_axis_limits_auto("x_axis")
    dpg.set_axis_limits_auto("y_axis")

def deploy_waveform():
    try:
        t_waveform, waveform, frequency = generate_waveform()
        rp = scpi(params["rp_ip"])
        rp.tx_txt("GEN:RST")
        rp.tx_txt("SOUR1:FUNC ARBITRARY")
        rp.tx_txt("SOUR1:TRAC:DATA:DATA " + ",".join(map(str, waveform)))
        rp.tx_txt("SOUR1:FREQ:FIX {}".format(frequency))
        rp.tx_txt("SOUR1:VOLT:OFFS 0")
        rp.tx_txt("OUTPUT1:STATE ON")
        rp.tx_txt("SOUR1:TRIG:INT")
        rp.close()
        dpg.set_value("status_text", "Waveform deployed to Red Pitaya")
    except OSError as e:
        dpg.set_value("status_text", "Error: Red Pitaya not reachable. Make sure it is properly connected, and wait 20-30s and retry.")
        print("OSError:", e)

def update_parameters():
    for key in params:
        if key == "rp_ip":
            params[key] = dpg.get_value(key)
        else:
            params[key] = float(dpg.get_value(key))
    plot_waveform()

def update_increments():
    for key in increments:
        increments[key] = float(dpg.get_value(f"inc_{key}"))
        dpg.configure_item(key, step=increments[key])

dpg.create_context()
dpg.set_global_font_scale(1.2)

with dpg.window(label="Waveform Generator", width=1000, height=800):
    with dpg.collapsing_header(label="Voltage Values"):
        with dpg.table(header_row=True):
            dpg.add_table_column(label="Parameter")
            dpg.add_table_column(label="Value")
            dpg.add_table_column(label="Increment")
            for key in ["v_start", "v_end", "v_down_end"]:
                with dpg.table_row():
                    dpg.add_text(key.replace("_", " ").title())
                    dpg.add_input_float(tag=key, default_value=params[key], step=increments[key], callback=update_parameters)
                    dpg.add_input_float(tag=f"inc_{key}", default_value=increments[key], callback=update_increments)
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
    dpg.add_input_text(label="Red Pitaya IP", tag="rp_ip", default_value=params["rp_ip"])
    dpg.add_button(label="Deploy to Red Pitaya", callback=deploy_waveform)
    dpg.add_text("", tag="status_text")
    with dpg.plot(label="Waveform Plot", height=300, width=500):
        dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="x_axis")
        dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude", tag="y_axis")
        dpg.add_line_series([], [], parent="y_axis", tag="waveform_plot")
        dpg.add_plot_legend()
        dpg.set_axis_limits_auto("x_axis")
        dpg.set_axis_limits_auto("y_axis")
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

