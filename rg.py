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
# TDGL Snapshot Real-Space RG Analysis (Option 1)
# Coarse-grain a field snapshot captured live from
# main.py and watch the statistics flow.
# =============================================

# Curie temperature of iron (K), matches main.py
TC = 1043.15


# Ensemble-averaged radial structure factor S(|k|)
def structure_factor(fields):
    N = fields[0].shape[0]
    power = np.zeros((N, N))
    for f in fields:
        fc = f - np.mean(f)                       # drop the k=0 mean
        power += np.abs(np.fft.fft2(fc)) ** 2     # accumulate power spectrum
    power /= len(fields) * N * N                  # ensemble + normalisation

    # Radial wavenumber grid
    k1d = 2.0 * np.pi * np.fft.fftfreq(N)         # wavenumbers per axis
    KX, KY = np.meshgrid(k1d, k1d)
    Kmag = np.sqrt(KX ** 2 + KY ** 2).ravel()

    # Bin the power into radial shells
    order = np.argsort(Kmag)
    k_sorted = Kmag[order]
    S_sorted = power.ravel()[order]
    nbins = N // 2
    edges = np.linspace(0, k_sorted.max(), nbins + 1)
    idx = np.digitize(k_sorted, edges)
    k_centers, S_radial = [], []
    for shell in range(1, nbins + 1):
        sel = idx == shell
        if np.any(sel):
            k_centers.append(k_sorted[sel].mean())
            S_radial.append(S_sorted[sel].mean())
    return np.array(k_centers), np.array(S_radial)


# Effective GL mass r and correlation length xi from S(k)
def effective_r_xi(fields):
    N = fields[0].shape[0]
    if N < 16:                                    # too few k-modes to fit
        return np.nan, np.nan
    k, S = structure_factor(fields)
    good = (k > 0) & (S > 0)
    k, S = k[good], S[good]
    if len(k) < 4:
        return np.nan, np.nan

    # Ornstein-Zernike: 1/S = (r + kappa*k^2)/C, linear in k^2
    n = max(4, len(k) // 4)                       # lowest-k portion only
    slope, intercept = np.polyfit(k[:n] ** 2, 1.0 / S[:n], 1)
    r_eff = intercept                             # inverse susceptibility (~ r)
    xi = np.sqrt(slope / intercept) if (slope > 0 and intercept > 0) else np.nan
    return r_eff, xi


# Coarse-grain a pre-built ensemble of fields and record the flow
def rg_field_flow_from_fields(fields, levels=5):
    fields = list(fields)
    snapshots = []      # one representative field per level
    records = []        # observables per level
    for _ in range(levels):
        size = fields[0].shape[0]
        r_eff, xi = effective_r_xi(fields)
        records.append({
            "size": size,                                          # lattice size
            "net_m": np.mean([f.mean() for f in fields]),          # net magnetization
            "order": np.mean([np.abs(f).mean() for f in fields]),  # local order |m|
            "var": np.mean([f.var() for f in fields]),             # field variance
            "r_eff": r_eff,                                        # effective GL mass
            "xi_over_L": xi / size if np.isfinite(xi) else np.nan, # corr length / L
        })
        snapshots.append(fields[0].copy())
        if size <= 4:                              # stop at the smallest block
            break
        fields = [renormalize(f) for f in fields]  # one RG step on every sample

    return records, snapshots


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
