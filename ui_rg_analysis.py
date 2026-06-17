# UI for RG Analysis 

#to run the UI, type "streamlit run app.py"

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Renormalization Group Analysis")

temperature = st.slider(
    "Temperature",
    0,
    1500,
    298
)
rg_step = st.slider(
    "RG Step",
    0,
    4,
    0
)

temperature_regime = st.selectbox(
    "Temperature Regime",
    [
        "Cold (298K)",
        "Critical (1043K)",
        "Hot (1500K)"
    ]
)
st.write("Current Temperature:", temperature)
st.write("Current RG Step:", rg_step)
st.write("Temperature Regime:", temperature_regime)

st.subheader("Lattice Visualization")

N = 128
lattice = 0.1 * np.random.randn(N, N)

fig, ax = plt.subplots()
ax.imshow(lattice, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_title("Initial TDGL Field / Lattice")
ax.set_xticks([])
ax.set_yticks([])

st.pyplot(fig)