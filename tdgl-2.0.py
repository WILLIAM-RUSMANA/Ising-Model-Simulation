import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from rg import renormalize

# ============================================
# Time-Dependent Ginzburg-Landau Simulation
# ============================================

# Grid size
N = 128

# Temperatures
Ti = 298.15    # Initial temperature (Kelvin)
Tc = 1043.15   # Critical temperature (Kelvin)

# Physical parameters
b = 1.0
kappa = 1.0
noise_intensity = 0.02  # Baseline strength of thermal fluctuations

# Time step
dt = 0.01

# Initialize magnetization field with small fluctuations
m = 0.1 * np.random.randn(N, N)

# --------------------------------------------
# Compute Laplacian using periodic boundaries
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
# Layout & Visualization Setup
# --------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(bottom=0.25)  # Make space at the bottom for the slider bar

text = ax.text(
    0.02,
    1.05,
    "",
    transform=ax.transAxes,
    fontsize=10,
    color="black",
    verticalalignment='top'
)

im = ax.imshow(m, cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest')

ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=0.1)

ax.set_title("Time Dependent GL Phase Transition Dynamics", y=1.08)

# --------------------------------------------
# Interactive Slider Setup (Fixed Colors & Alignment)
# --------------------------------------------
ax_temp = fig.add_axes([0.2, 0.08, 0.6, 0.03])  # [left, bottom, width, height]

slider_temp = Slider(
    ax=ax_temp,
    label='Temperature (K) ',
    valmin=0.0,
    valmax=2000.0,     # The range of your temperature domain
    valinit=Ti,        # Starts at room temperature (298.15 K)
    valfmt='%0.1f K',
    color='skyblue'    # Colors the filled track light blue
)

# FORCE the slider's internal axes to match the 0-2000 data limits
ax_temp.set_xlim(0.0, 2000.0)

slider_temp.poly.set_facecolor('deepskyblue')  # Sets the knob face color
slider_temp.poly.set_edgecolor('skyblue')      # Sets the knob border color

# vertical indicator line
ax_temp.axvline(x=Tc, color='red', linestyle='--', linewidth=1.5)

# red TC text above bar
ax_temp.text(Tc, 1.2, 'Tc (1043 K)', color='red', fontsize=9, ha='center', transform=ax_temp.get_xaxis_transform())


# --------------------------------------------
# Renormalization Group Real-Space Flow Tracker
# --------------------------------------------
def rg_flow(field, levels=5):
    current = field.copy()
    sizes = []
    mags = []

    for _ in range(levels):
        N_curr = current.shape[0]
        sizes.append(N_curr)
        mags.append(np.mean(np.abs(current)))

        if N_curr <= 2:
            break

        current = renormalize(current)

    return sizes, mags

# Tracking steps
frame_count = 0

# --------------------------------------------
# Time Evolution Loop
# --------------------------------------------
def update(_):
    global m, frame_count
    frame_count += 1
    
    # 1. Read temperature dynamically from the interactive slider
    t = slider_temp.val
    
    # 2. Compute the temperature dependent linear mass coefficient
    effective_a = (Tc - t) / Tc   

    # 3. Compute Langevin Thermal Noise (Fluctuation-Dissipation theorem)
    # The noise variance scales with temperature T, vanishing at T = 0K
    if t > 0:
        noise_scale = np.sqrt(2.0 * (t / Tc) * noise_intensity * dt)
        thermal_noise = np.random.normal(0, noise_scale, size=(N, N))
    else:
        thermal_noise = 0.0

    # 4. Deterministic TDGL forces
    dm_dt = effective_a * m - b * m**3 + kappa * laplacian(m)

    # 5. Integrate using Semi-Stochastic Euler update
    m += dt * dm_dt + thermal_noise

    # 6. Periodic RG tracking (Decoupled from temperature values to avoid console lag)
    if frame_count % 50 == 0:
        sizes, mags = rg_flow(m)
        print(f"\n--- [Frame {frame_count}] RG FLOW ANALYSIS (T = {t:.2f}K) ---")
        for s, mag in zip(sizes, mags):
            print(f"{s}x{s} | |M| = {mag:.4f}")

    # 7. Update display elements
    im.set_array(m)

    avg_magnetization = np.mean(m)
    abs_magnetization = np.mean(np.abs(m))
    phase = "Ferromagnetic (Ordered)" if t < Tc else "Paramagnetic (Disordered)"

    text.set_text(
        f"T = {t:.2f}K | Phase = {phase}\nMagnetization = {avg_magnetization:.3f} | Absolute M = {abs_magnetization:.3f}"
    )

    return [im, text]

# Note: blit=False is used to ensure the slider graphics redraw smoothly during dragging
ani = FuncAnimation(fig, update, interval=20, blit=False)

plt.show()