import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from rg import IsingRG, renormalize

# ============================================
# Time-Dependent Ginzburg-Landau Simulation
# ============================================

# Folder where live field snapshots are saved for the Streamlit RG app to read
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rg_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Grid size
N = 128

# Temperatures
Ti = 298.15    # Initial temperature (Kelvin)
Tc = 1043.15   # Critical temperature of Iron (Kelvin)

# Physical parameters
b = 1.0
kappa = 1.0
noise_intensity = 0.02  # Baseline strength of thermal fluctuations

# Time step
dt = 0.01
sim_speed_multiplier = 1  # Default math loops per frame step
is_paused = False         # Pause state flag
snapshot_flash_frames = 0 # Countdown for the "Saved!" button label

# Initialize magnetization field with small fluctuations
m = 0.1 * np.random.randn(N, N)

# Arrays to keep a rolling history for the separate graph page
history_frames = []
history_mag = []
history_abs_mag = []
max_history_len = 300  
frame_count = 0

# --------------------------------------------
# Discrete Laplacian Operator
# --------------------------------------------
def laplacian(field):
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4 * field
    )

# --------------------------------------------
# WINDOW 1: Spatial Grid Map Setup
# --------------------------------------------
fig_sim = plt.figure("Phase Grid Map", figsize=(6.5, 7.0))
ax_sim = fig_sim.add_axes([0.15, 0.24, 0.75, 0.56]) # Adjusted to allow button space underneath

# ADJUST TOP SPACING: Shrinks viewport from the top down to prevent clipping completely
fig_sim.subplots_adjust(top=0.78)

im = ax_sim.imshow(m, cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest')

# Placed the main title high up on the axis frame
ax_sim.set_title("Time Dependent GL Phase Transition Dynamics", fontsize=12, fontweight='bold', pad=42)

# Readjusted metadata lines: Placed cleanly above the grid without text overlaps
text_stats = ax_sim.text(
    0.0, 1.04, "", 
    transform=ax_sim.transAxes, 
    fontsize=9.5, 
    linespacing=1.4,
    verticalalignment='bottom',
    horizontalalignment='left'
)

# --------------------------------------------
# WINDOW 2: Separate Page Line Graph Setup
# --------------------------------------------
fig_graph = plt.figure("Thermodynamic Metrics Page", figsize=(7, 4.5))
ax_graph = fig_graph.add_subplot(111)

line_mag, = ax_graph.plot([], [], label='Net Magnetization ($M$)', color='darkorange', linewidth=1.5)
line_abs_mag, = ax_graph.plot([], [], label='Abs Magnetization ($|M|$)', color='purple', linewidth=1.5)
ax_graph.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax_graph.set_title("Real-Time Order Parameter Tracking", fontsize=11)
ax_graph.set_xlabel("Simulation Frames")
ax_graph.set_ylabel("Magnetization Intensity")
ax_graph.legend(loc='upper right', fontsize=9)
ax_graph.grid(True, linestyle=':', alpha=0.6)

# --------------------------------------------
# Widgets Setup (On Window 1)
# --------------------------------------------
# Temperature Slider
ax_temp = fig_sim.add_axes([0.25, 0.14, 0.55, 0.03])
slider_temp = Slider(
    ax=ax_temp, label='Temperature (K) ',
    valmin=0.0, valmax=2000.0, valinit=Ti,
    valfmt='%0.1f K', color='skyblue'
)
ax_temp.set_xlim(0.0, 2000.0)
slider_temp.poly.set_facecolor('deepskyblue')
slider_temp.poly.set_edgecolor('skyblue')

# Draw the critical temperature target line
ax_temp.axvline(x=Tc, color='red', linestyle='--', linewidth=1.5)
ax_temp.text(Tc, 1.3, 'Tc (1043 K)', color='red', fontsize=9, ha='center', transform=ax_temp.get_xaxis_transform())

# Simulation Control Interactive Buttons
ax_b1 = fig_sim.add_axes([0.15, 0.05, 0.13, 0.04])
ax_b2 = fig_sim.add_axes([0.31, 0.05, 0.13, 0.04])
ax_b3 = fig_sim.add_axes([0.47, 0.05, 0.13, 0.04])
ax_bp = fig_sim.add_axes([0.68, 0.05, 0.17, 0.04]) # Pause button
ax_bsnap = fig_sim.add_axes([0.15, 0.005, 0.70, 0.035]) # Snapshot button

btn_1x = Button(ax_b1, '1x Speed', color='gainsboro', hovercolor='lightgray')
btn_2x = Button(ax_b2, '2x Speed', color='gainsboro', hovercolor='lightgray')
btn_5x = Button(ax_b3, '5x Speed', color='gainsboro', hovercolor='lightgray')
btn_pause = Button(ax_bp, 'Pause', color='gainsboro', hovercolor='lightgray')
btn_snap = Button(ax_bsnap, 'Snapshot for RG', color='lightyellow', hovercolor='khaki')

def set_speed_1x(event): global sim_speed_multiplier; sim_speed_multiplier = 1
def set_speed_2x(event): global sim_speed_multiplier; sim_speed_multiplier = 2
def set_speed_5x(event): global sim_speed_multiplier; sim_speed_multiplier = 5

def toggle_pause(event):
    global is_paused
    is_paused = not is_paused
    if is_paused:
        btn_pause.label.set_text('Resume')
        btn_pause.color = 'palegreen'
    else:
        btn_pause.label.set_text('Pause')
        btn_pause.color = 'gainsboro'
    fig_sim.canvas.draw_idle()

def save_snapshot(event):
    global snapshot_flash_frames
    t = slider_temp.val
    now = time.time()
    # Filename starts with a sortable timestamp; T/frame are just for browsing
    stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(now))
    ms = int(now * 1000) % 1000
    fname = f"{stamp}_{ms:03d}_T{t:.0f}K_f{frame_count}.npz"
    np.savez(os.path.join(SNAPSHOT_DIR, fname), field=m, temperature=t,
             frame_count=frame_count, timestamp=now)
    btn_snap.label.set_text('Saved!')
    snapshot_flash_frames = 40  # ~0.8s at 20ms/frame before reverting
    fig_sim.canvas.draw_idle()

btn_1x.on_clicked(set_speed_1x)
btn_2x.on_clicked(set_speed_2x)
btn_5x.on_clicked(set_speed_5x)
btn_pause.on_clicked(toggle_pause)
btn_snap.on_clicked(save_snapshot)

# --------------------------------------------
# Renormalization Group Coarse-Graining 
# --------------------------------------------
def rg_flow(field, levels=5):
    current = field.copy()
    sizes, mags = [], []
    for _ in range(levels):
        N_curr = current.shape[0]
        sizes.append(N_curr)
        mags.append(np.mean(np.abs(current)))
        if N_curr <= 2:
            break
        current = renormalize(current)
    return sizes, mags

# --------------------------------------------
# Time Evolution Animation Loop
# --------------------------------------------
def update(_):
    global m, frame_count, snapshot_flash_frames

    if snapshot_flash_frames > 0:
        snapshot_flash_frames -= 1
        if snapshot_flash_frames == 0:
            btn_snap.label.set_text('Snapshot for RG')

    t = slider_temp.val
    phase = "Ferromagnetic (Ordered)" if t < Tc else "Paramagnetic (Disordered)"
    
    # If paused, bypass physical integration changes completely but preserve static frame rendering
    if not is_paused:
        effective_a = (Tc - t) / Tc   

        # Execute multiple math integration steps per render frame based on button choices
        for _ in range(sim_speed_multiplier):
            frame_count += 1
            
            if t > 0:
                noise_scale = np.sqrt(2.0 * (t / Tc) * noise_intensity * dt)
                thermal_noise = np.random.normal(0, noise_scale, size=(N, N))
            else:
                thermal_noise = 0.0

            # Time Dependent Ginzburg Landau Equation Calculation
            dm_dt = effective_a * m - b * m**3 + kappa * laplacian(m)
            # Euler-Maruyama method
            m += dt * dm_dt + thermal_noise 

        # Spatial Thermodynamics
        avg_magnetization = np.mean(m)
        abs_magnetization = np.mean(np.abs(m))

        # Append to rolling timeline data arrays
        history_frames.append(frame_count)
        history_mag.append(avg_magnetization)
        history_abs_mag.append(abs_magnetization)

        if len(history_frames) > max_history_len:
            history_frames.pop(0)
            history_mag.pop(0)
            history_abs_mag.pop(0)

        # Print out console diagnostics 
        if frame_count % (50 * sim_speed_multiplier) < sim_speed_multiplier:
            sizes, mags = rg_flow(m)
            print(f"\n--- [Frame {frame_count}] RG ANALYSIS (Speed: {sim_speed_multiplier}x) ---")
            for s, mag in zip(sizes, mags):
                print(f"{s}x{s} | |M| = {mag:.4f}")

        # Render Updates (Window 1)
        im.set_array(m)
        text_stats.set_text(
            f"T = {t:.2f}K | Phase = {phase} | Calculation Pace = {sim_speed_multiplier}x\n"
            f"Avg M = {avg_magnetization:.3f} | Abs Order |M| = {abs_magnetization:.3f}"
        )

        # Render Updates (Window 2 - Separate Page)
        line_mag.set_data(history_frames, history_mag)
        line_abs_mag.set_data(history_frames, history_abs_mag)
        
        ax_graph.set_xlim(history_frames[0], history_frames[-1] + 5)
        ax_graph.set_ylim(-1.1, 1.1)
    else:
        # Update text to display frozen state status if paused
        text_stats.set_text(
            f"T = {t:.2f}K | Phase = {phase} | State = PAUSED\n"
            f"Avg M = {np.mean(m):.3f} | Abs Order |M| = {np.mean(np.abs(m)):.3f}"
        )

    # Draw both figures during the loop execution safely
    fig_sim.canvas.draw_idle()
    fig_graph.canvas.draw_idle()

    return [im, text_stats, line_mag, line_abs_mag]

# Attach tracking animation globally
ani = FuncAnimation(fig_sim, update, interval=20, blit=False)
plt.show()