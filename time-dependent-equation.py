import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from rg import IsingRG, renormalize

# ============================================
# Time-Dependent Ginzburg-Landau Simulation
# ============================================

# Grid size
N = 128

# Temperatures
Ti = 298.15    # initial temp
t = Ti         # temperature
Tc = 1043.15   # Critical temp

# Physical parameters
b = 1.0
kappa = 1.0

# Time step
dt = 0.01

# --------------------------------------------
# Initialize magnetization field
# Small random fluctuations around zero
# --------------------------------------------

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
# Time evolution
# --------------------------------------------

fig, ax = plt.subplots()
text = ax.text(
    0.02,
    1.07,
    "",
    transform=ax.transAxes,
    fontsize=10,
    color="black",
    verticalalignment='top'
)

# im = ax.imshow(m, cmap='coolwarm', vmin=-1, vmax=1)
im = ax.imshow(m, cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest')

ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=0.1)

ax.set_title("TDGL Phase Transition Dynamics", y=1.08)

def rg_flow(field, levels=5):
    current = field.copy()
    sizes = []
    mags = []

    for _ in range(levels):
        N = current.shape[0]
        sizes.append(N)
        mags.append(np.mean(np.abs(current)))

        if N <= 2:
            break

        current = renormalize(current)

    return sizes, mags

def update(_):  # _: A frame parameter is automatically passed by FuncAnimation
    global t, m
    speed = 1.5 - 1.4 * np.exp(-((t - Tc)/80)**2)
    t += speed
    # TDGL equation
    effective_a = (Tc - t) / Tc   # normalized

    dm_dt = effective_a * m - b * m**3 + kappa * laplacian(m)

    # Euler update
    m += dt * dm_dt
    if int(t) % 50 == 0:
        sizes, mags = rg_flow(m)

        print("\nRG FLOW")
        for s, mag in zip(sizes, mags):
            print(f"{s}x{s} | |M| = {mag:.4f}")

    # Update image
    im.set_array(m)

    avg_magnetization = np.mean(m)
    abs_magnetization = np.mean(np.abs(m))

    if t < Tc:
        phase = "Ferromagnetic"
    else:
        phase = "Paramagnetic"

    text.set_text(
        f"T = {t:.2f}K | Phase = {phase} | Magnetization = {avg_magnetization:.3f} | Absolute M = {abs_magnetization}"
    )

    return [im, text]

ani = FuncAnimation(fig, update, interval=20)

plt.show()