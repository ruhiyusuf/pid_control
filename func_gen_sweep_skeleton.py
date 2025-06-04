# Updated and deduplicated script version based on user's instructions

import numpy as np
import time
import threading
import dearpygui.dearpygui as dpg
from redpitaya_scpi import scpi  # Required for deploy_waveform()

params = {
    "v_start": 0.0,
    "v_end": 1.0,
    "v_down_end": 0.8,
    "start_v_ms": 20.0,
    "slope_up_ms": 20.0,
    "slope_down_ms": 40.0,
    "end_v_ms": 5000.0,
    "rp_ip": "rp-f0cbc6.local",
    "output_channel": 1
}

waveform1_params = params.copy()
waveform2_params = params.copy()
y_min, y_max = -0.2, 1.2
sweep_duration = 5.0

sweep_running = False
sweep_paused = False
loop_enabled = False

def generate_waveform(p=None, num_samples=16384):
    p = p or params
    start = p["start_v_ms"] / 1000
    up = p["slope_up_ms"] / 1000
    down = p["slope_down_ms"] / 1000
    end = p["end_v_ms"] / 1000
    total_time = start + up + down + end
    frequency = 1 / total_time
    t = np.linspace(0, total_time, num_samples)

    waveform = np.piecewise(t,
        [t < start,
         (t >= start) & (t < start + up),
         (t >= start + up) & (t < start + up + down),
         t >= start + up + down],
        [lambda t: p["v_start"],
         lambda t: p["v_start"] + (p["v_end"] - p["v_start"]) * (t - start) / up,
         lambda t: p["v_end"] + (p["v_down_end"] - p["v_end"]) * (t - start - up) / down,
         lambda t: p["v_down_end"]])
    return t, waveform, frequency

def sweep_waveform_gui_only(w1, w2, duration):
    global sweep_running, sweep_paused, loop_enabled
    sweep_running = True
    while sweep_running:
        start_time = time.perf_counter()
        while True:
            if sweep_paused:
                time.sleep(0.05)
                continue
            alpha = min((time.perf_counter() - start_time) / duration, 1.0)
            waveform = (1 - alpha) * w1 + alpha * w2
            dpg.set_value("sweep_plot_series", [list(range(len(waveform))), waveform.tolist()])
            if alpha >= 1.0:
                break
            time.sleep(0.02)
        if not loop_enabled:
            break
        w1, w2 = w2, w1
    sweep_running = False

def toggle_sweep():
    global sweep_paused
    sweep_paused = not sweep_paused
    dpg.set_value("status_text", "Paused" if sweep_paused else "Running")

def start_preview():
    global sweep_running, loop_enabled
    if sweep_running:
        return

    # Sync updated values from GUI inputs
    sync_gui_params()

    loop_enabled = dpg.get_value("loop_toggle")
    t, w1, _ = generate_waveform(waveform1_params)
    _, w2, _ = generate_waveform(waveform2_params)
    print("[START SWEEP] max w1 =", max(w1), "| max w2 =", max(w2))
    print("[START SWEEP] min w1 =", min(w1), "| min w2 =", min(w2))

    thread = threading.Thread(
        target=sweep_waveform_with_output,
        args=(t, np.array(w1), np.array(w2), sweep_duration)
    )
    thread.start()

def sync_gui_params():
    global waveform1_params, waveform2_params
    for key in params:
        if key in ["rp_ip", "output_channel"]:
            continue
        val1 = dpg.get_value(f"{key}_1")
        val2 = dpg.get_value(f"{key}_2")

        waveform1_params[key] = float(val1) if val1 is not None else params[key]
        waveform2_params[key] = float(val2) if val2 is not None else params[key]

def start_sweep_to_redpitaya():
    global sweep_running, loop_enabled
    if sweep_running:
        return

    sync_gui_params()
    loop_enabled = False  # One-shot deployment

    print("[START] Synced waveform1_params:", waveform1_params)
    print("[START] Synced waveform2_params:", waveform2_params)

    t, w1, _ = generate_waveform(waveform1_params)
    _, w2, _ = generate_waveform(waveform2_params)

    print("[START] max w1 =", max(w1), "min w1 =", min(w1))
    print("[START] max w2 =", max(w2), "min w2 =", min(w2))

    # comment later
    loop_enabled = False
    thread = threading.Thread(
        target=sweep_waveform_with_output,
        args=(t, np.array(w1), np.array(w2), sweep_duration)
    )
    thread.start()

def stop_sweep():
    global sweep_running
    sweep_running = False
    dpg.set_value("status_text", "Stopped")

def update_waveform(tab_id):
    global waveform1_params, waveform2_params
    for key in params:
        if key in ["rp_ip", "output_channel"]:
            continue
        val = dpg.get_value(f"{key}_{tab_id}")
        if tab_id == 1:
            waveform1_params[key] = float(val)
        else:
            waveform2_params[key] = float(val)
    t, y, _ = generate_waveform(waveform1_params if tab_id == 1 else waveform2_params)
    tag = "plot1_series" if tab_id == 1 else "plot2_series"
    dpg.set_value(tag, [t, y])

def add_inputs(tab_id, values):
    for key in params:
        if key in ["rp_ip", "output_channel"]:
            continue
        dpg.add_input_float(label=key, default_value=values[key], tag=f"{key}_{tab_id}",
                            callback=lambda s, a, u: update_waveform(tab_id))

def deploy_waveform(waveform=None, frequency=None):
    try:
        if waveform is None or frequency is None:
            raise ValueError("Waveform and frequency must be explicitly passed to deploy_waveform() during sweep.")

        rp = scpi(params["rp_ip"])
        ch = int(params["output_channel"])
        assert ch in [1, 2], "Output channel must be 1 or 2"
        rp.tx_txt(f"SOUR{ch}:FUNC ARBITRARY")
        rp.tx_txt(f"SOUR{ch}:TRAC:DATA:DATA " + ",".join(map(str, waveform)))
        rp.tx_txt(f"SOUR{ch}:FREQ:FIX {frequency}")
        rp.tx_txt(f"SOUR{ch}:VOLT:OFFS 0")
        rp.tx_txt(f"OUTPUT{ch}:STATE ON")
        rp.tx_txt(f"SOUR{ch}:TRIG:INT")
        # rp.close()
        print(f"[DEPLOY] Sent to Red Pitaya, Freq = {frequency:.4f} Hz")
    except Exception as e:
        print("[DEPLOY ERROR]", e)
        dpg.set_value("status_text", "Red Pitaya Error: " + str(e))

def sweep_waveform_gui_only(t, w1, w2, duration):
    global sweep_running, sweep_paused, loop_enabled
    sweep_running = True
    while sweep_running:
        start_time = time.perf_counter()
        while True:
            if sweep_paused:
                time.sleep(0.05)
                continue
            alpha = min((time.perf_counter() - start_time) / duration, 1.0)
            waveform = (1 - alpha) * w1 + alpha * w2
            print(f"[SWEEP] alpha={alpha:.2f} | max={np.max(waveform):.3f} | min={np.min(waveform):.3f}")
            dpg.set_value("sweep_plot_series", [t.tolist(), waveform.tolist()])
            if alpha >= 1.0:
                break
            time.sleep(0.02)
        if not loop_enabled:
            break
        w1, w2 = w2, w1
    sweep_running = False

def sweep_waveform_with_output(t, w1, w2, duration, update_interval=0.02):
    global sweep_running, sweep_paused, loop_enabled
    sweep_running = True

    while sweep_running:
        start_time = time.perf_counter()

        while True:
            if sweep_paused:
                time.sleep(0.05)
                continue

            alpha = min((time.perf_counter() - start_time) / duration, 1.0)
            waveform = (1 - alpha) * w1 + alpha * w2
            frequency = 1 / (t[-1] - t[0])

            # Update GUI
            dpg.set_value("sweep_plot_series", [t.tolist(), waveform.tolist()])

            # Deploy to RP
            deploy_waveform(waveform.tolist(), frequency)

            if alpha >= 1.0:
                break

            time.sleep(update_interval)

        if not loop_enabled:
            break
        w1, w2 = w2, w1

    sweep_running = False

def build_gui():
    dpg.create_context()
    with dpg.window(label="Waveform Sweep Preview", width=1000, height=800):
        # dpg.add_button(label="Start Preview", callback=start_preview)
        dpg.add_button(label="Pause/Resume", callback=toggle_sweep)
        dpg.add_button(label="Stop", callback=stop_sweep)
        dpg.add_button(label="Deploy to Red Pitaya", callback=start_sweep_to_redpitaya)
        dpg.add_checkbox(label="Loop Sweep", tag="loop_toggle")
        dpg.add_text("", tag="status_text")

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

            with dpg.tab(label="Sweep Preview"):
                with dpg.plot(label="Sweeping", height=300, width=900):
                    dpg.add_plot_axis(dpg.mvXAxis)
                    dpg.add_plot_axis(dpg.mvYAxis, tag="axis_sweep")
                    dpg.add_line_series([], [], parent="axis_sweep", tag="sweep_plot_series")
                    dpg.set_axis_limits("axis_sweep", y_min, y_max)

    dpg.create_viewport(title='Waveform Sweep Preview', width=1000, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    update_waveform(1)
    update_waveform(2)
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    build_gui()


