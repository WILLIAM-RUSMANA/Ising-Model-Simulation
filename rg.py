import numpy as np
import matplotlib.pyplot as plt

# =========================
# 2D Ising Model + Real-Space Renormalization Group
# =========================

class IsingRG:
    def __init__(self, size=64, temperature=2.0, J=1.0):
        self.size = size
        self.T = temperature
        self.J = J

        # Random initial lattice of spins (+1 or -1)
        self.lattice = np.random.choice([-1, 1], size=(size, size))

    # ---------------------------------------------
    # Plot RG flow
    # ---------------------------------------------
    def plot_rg_flow(self, sizes, mags, energies):

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(sizes, mags, marker='o')
        plt.gca().invert_xaxis()
        plt.xlabel("Lattice Size")
        plt.ylabel("Magnetization")
        plt.title("RG Flow of Magnetization")

        plt.subplot(1, 2, 2)
        plt.plot(sizes, energies, marker='o')
        plt.gca().invert_xaxis()
        plt.xlabel("Lattice Size")
        plt.ylabel("Energy per Spin")
        plt.title("RG Flow of Energy")

        plt.tight_layout()
        plt.show()

# ---------------------------------------------
# Compute magnetization
# ---------------------------------------------
def magnetization(self, lattice=None):
    if lattice is None:
        lattice = self.lattice

    return np.sum(lattice) / lattice.size

# ---------------------------------------------
# Real-space RG block transformation
# Majority rule on 2x2 blocks
# ---------------------------------------------
def renormalize(field):

    N = field.shape[0]

    if N % 2 != 0:
        raise ValueError("Field size must be even.")

    coarse = 0.25 * (
        field[0::2, 0::2]
        + field[1::2, 0::2]
        + field[0::2, 1::2]
        + field[1::2, 1::2]
    )

    return coarse

# ---------------------------------------------
# Compute nearest-neighbor energy
# ---------------------------------------------
def energy(self, lattice=None):
    if lattice is None:
        lattice = self.lattice

    E = 0
    N = lattice.shape[0]

    for i in range(N):
        for j in range(N):

            S = lattice[i, j]

            # Periodic boundary conditions
            right = lattice[i, (j + 1) % N]
            down = lattice[(i + 1) % N, j]

            E += -self.J * S * (right + down)

    return E / lattice.size


# =========================
# Testing
# =========================
if __name__ == "__main__":
    model = IsingRG(size=64, temperature=2.2)

    sizes, mags, energies = model.run_rg(iterations=6)

    model.plot_rg_flow(sizes, mags, energies)