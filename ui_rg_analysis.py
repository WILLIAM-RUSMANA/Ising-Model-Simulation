# Streamlit UI for TDGL Real-Space RG Analysis (Option 1)
# Run with:  streamlit run ui_rg_analysis.py

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from rg import rg_field_flow_from_fields, TC

# Folder where main.py saves live field snapshots for this app to read
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rg_snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

st.set_page_config(page_title="RG Analysis", layout="wide")

# Widen the sidebar so the snapshot dropdown labels aren't cut off
st.markdown(
    "<style>section[data-testid='stSidebar'] { width: 340px !important; }</style>",
    unsafe_allow_html=True,
)

st.title("Real-Space Renormalization Group Analysis")

# Sidebar controls
st.sidebar.header("Controls")

raw_files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith(".npz")]
if not raw_files:
    st.warning(
        "No snapshots found. Run main.py, click **Snapshot for RG** one or "
        "more times, then refresh this page."
    )
    st.stop()


# Read each snapshot's embedded metadata (not the filename) for sorting/labels
def read_meta(fname):
    with np.load(os.path.join(SNAPSHOT_DIR, fname)) as d:
        return float(d["temperature"]), int(d["frame_count"]), float(d["timestamp"])


meta = {f: read_meta(f) for f in raw_files}
snapshot_files = sorted(meta, key=lambda f: meta[f][2], reverse=True)  # newest first
labels = {
    f: f"{datetime.fromtimestamp(meta[f][2]).strftime('%m-%d %H:%M:%S')}  "
       f"T={meta[f][0]:.0f}K  f{meta[f][1]}"
    for f in snapshot_files
}

chosen = st.sidebar.selectbox("Snapshot", snapshot_files, format_func=lambda f: labels[f])
levels = st.sidebar.slider("RG levels", 2, 6, 5)

# Load the chosen field itself
with np.load(os.path.join(SNAPSHOT_DIR, chosen)) as data:
    snap_field = data["field"]
temperature, snap_frame, snap_time = meta[chosen]
saved_at = datetime.fromtimestamp(snap_time).strftime("%H:%M:%S")

st.info(f"Snapshot — T = {temperature:.1f} K, frame {snap_frame}, saved {saved_at}")

records, snapshots = rg_field_flow_from_fields([snap_field], levels=levels)

# Regime label from temperature
a = (TC - temperature) / TC                       # GL control parameter
if temperature < TC - 60:
    regime = "Ferromagnetic (ordered), a > 0"
elif temperature > TC + 60:
    regime = "Paramagnetic (disordered), a < 0"
else:
    regime = "Near-critical, a ≈ 0"
st.write(f"**T = {temperature:.1f} K**  |  Tc = {TC:.0f} K  |  {regime}")

# --- Field snapshots at each RG level ---
st.subheader("Coarse-graining the field")
vmax = np.abs(snapshots[0]).max() or 1.0          # shared colour scale
fig_fields, axes = plt.subplots(1, len(snapshots),
                                figsize=(2.4 * len(snapshots), 2.8))
for ax, snap in zip(np.atleast_1d(axes), snapshots):
    img = ax.imshow(snap, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                    interpolation="nearest")
    ax.set_title(f"{snap.shape[0]}x{snap.shape[0]}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig_fields.colorbar(img, ax=axes, fraction=0.025, pad=0.02)
st.pyplot(fig_fields)

# --- RG flow of the observables ---
st.subheader("RG flow of the statistics")

# Pull each observable into its own list
sizes = [r["size"] for r in records]
xs = list(range(len(records)))
order = [r["order"] for r in records]
variance = [r["var"] for r in records]
r_eff = [r["r_eff"] for r in records]
xi_over_L = [r["xi_over_L"] for r in records]


# Draw an "all NaN" note when an observable is undefined
def note_if_empty(ax, values):
    if all(np.isnan(v) for v in values):
        ax.text(0.5, 0.5, "not available\n(ordered phase / lattice too small)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)


fig_flow, ax = plt.subplots(2, 2, figsize=(9, 6))

# Local order
ax[0, 0].plot(xs, order, "o-", color="crimson")
ax[0, 0].set_title("Local order |m|")
ax[0, 0].set_ylim(0, 1)

# Field variance
ax[0, 1].plot(xs, variance, "o-", color="darkorange")
ax[0, 1].set_title("Field variance")

# Effective GL mass
ax[1, 0].plot(xs, r_eff, "o-", color="royalblue")
ax[1, 0].axhline(0, color="gray", ls="--", lw=0.8)
ax[1, 0].set_yscale("symlog")
ax[1, 0].set_title("Effective GL mass r  (~ 1/S(0))")
note_if_empty(ax[1, 0], r_eff)

# Correlation length over system size
ax[1, 1].plot(xs, xi_over_L, "o-", color="seagreen")
ax[1, 1].set_title("Correlation length / L")
note_if_empty(ax[1, 1], xi_over_L)

# Shared x-axis: RG step labelled by lattice size
for sub in ax.ravel():
    sub.set_xticks(xs)
    sub.set_xticklabels(sizes)
    sub.set_xlabel("lattice size  (coarse-graining →)")
    sub.grid(True, ls=":", alpha=0.6)
fig_flow.tight_layout()
st.pyplot(fig_flow)

# --- Plain-language verdict from the flow ---
st.subheader("What the flow means")
order0, orderL = order[0], order[-1]
if order0 > 0.4 and orderL > 0.4:
    verdict = (
        "Order survives coarse-graining and stays nearly scale-invariant — the "
        "field is flowing to the **ordered (ferromagnetic) fixed point**."
    )
elif orderL < 0.6 * order0:
    verdict = (
        "Order washes out as you coarse-grain (short-range fluctuations average "
        "to zero) — the field is flowing to the **disordered (paramagnetic) "
        "fixed point**."
    )
else:
    verdict = (
        "Order changes only slowly under coarse-graining — the field sits near "
        "the **unstable critical fixed point**, the slowest point of the flow."
    )
st.markdown(verdict)

# Raw per-level numbers (renamed for display only)
column_labels = {
    "order": "order (abs. magnetization)",
    "net_m": "net_m (net magnetization)",
    "var": "var (field variance)",
    "r_eff": "r_eff (effective GL mass)",
    "xi_over_L": "xi_over_L (corr. length / L)",
}
display_records = [
    {column_labels.get(k, k): v for k, v in r.items()}
    for r in records
]
st.dataframe(display_records, width="stretch")

