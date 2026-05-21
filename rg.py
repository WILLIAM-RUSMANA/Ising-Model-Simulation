import numpy as np
import matplotlib.pyplot as plt

# =============================================
# 2D Ising Model + Real-Space Renormalization Group
# =============================================

class IsingRG:
    def __init__(self, size=64, temperature=2.2, J=1.0):
        self.size = size
        self.T = temperature
        self.J = J

        # Random initial lattice of spins (+1 or -1)
        self.lattice = np.random.choice([-1, 1], size=(size, size))

    def magnetization(self, lattice=None):
        """Compute the average magnetization of the given lattice."""
        if lattice is None:
            lattice = self.lattice
        return np.sum(lattice) / lattice.size

    def energy(self, lattice=None):
        """Compute nearest-neighbor energy per spin with periodic boundary conditions."""
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

    def run_rg(self, iterations=5):
        """
        Runs the real-space renormalization group transformations over multiple 
        iterations, saving the lattice size, magnetization, and energy at each step.
        """
        sizes = []
        mags = []
        energies = []

        current_lattice = self.lattice.copy()

        for i in range(iterations):
            N = current_lattice.shape[0]
            sizes.append(N)
            
            # Record metrics BEFORE collapsing the lattice further
            mags.append(np.abs(self.magnetization(current_lattice)))
            energies.append(self.energy(current_lattice))

            if N <= 2:
                print(f"Stopping RG at iteration {i}: Minimal lattice size reached.")
                break
                
            if N % 2 != 0:
                print(f"Stopping RG at iteration {i}: Lattice size {N} is not divisible by 2.")
                break

            # Apply the block spin transformation
            current_lattice = renormalize(current_lattice)

        return sizes, mags, energies

    def plot_rg_flow(self, sizes, mags, energies):
        """Plot the RG flow of magnetization and energy."""
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(sizes, mags, marker='o', color='crimson')
        plt.gca().invert_xaxis()  # RG flows from large systems to small systems (coarse-graining)
        plt.xlabel("Lattice Size")
        plt.ylabel("Absolute Magnetization")
        plt.title("RG Flow of Magnetization")

        plt.subplot(1, 2, 2)
        plt.plot(sizes, energies, marker='o', color='royalblue')
        plt.gca().invert_xaxis()
        plt.xlabel("Lattice Size")
        plt.ylabel("Energy per Spin")
        plt.title("RG Flow of Energy")

        plt.tight_layout()
        plt.show()

# ---------------------------------------------
# Real-space RG block transformation
# Majority rule on 2x2 blocks (Kept global for easy external importing)
# ---------------------------------------------
def renormalize(field):
    N = field.shape[0]

    if N % 2 != 0:
        raise ValueError("Field size must be even to perform 2x2 block renormalization.")

    # Using sign() handles majority rule for spins (+1 vs -1) 
    # and handles continuous values nicely for your TDGL field script.
    coarse = 0.25 * (
        field[0::2, 0::2]
        + field[1::2, 0::2]
        + field[0::2, 1::2]
        + field[1::2, 1::2]
    )

    # If working strictly with discrete spins (+1, -1), you can un-comment the next line:
    # return np.sign(coarse)
    return coarse


# =============================================
# Testing
# =============================================
if __name__ == "__main__":
    # Initialize the model (Note: size must be a power of 2 for multiple clean reductions)
    model = IsingRG(size=64, temperature=2.2)

    # Run the RG trajectory
    sizes, mags, energies = model.run_rg(iterations=5)

    # Render results
    model.plot_rg_flow(sizes, mags, energies)