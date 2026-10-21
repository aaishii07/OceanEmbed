from pathlib import Path
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="OceanEmbed", page_icon="🌊")
st.title("🌊 OceanEmbed")
st.caption("Subsurface temperature reconstruction from surface ocean patches")
runs = sorted(Path("results").glob("*/predictions.npz"))
if not runs:
    st.info("Train a model first. The demo will discover results/<run>/predictions.npz automatically.")
    st.code("python -m src.oceanembed.training.train --data data/processed/oceanembed_mvp.npz")
    st.stop()
selected = st.selectbox("Experiment", runs, format_func=lambda p: p.parent.name)
data = np.load(selected); row = st.slider("Validation sample", 0, len(data["observed"]) - 1, 0)
fig, ax = plt.subplots(figsize=(5, 6))
ax.plot(data["observed"][row], data["depths_m"], "o-", label="GLORYS target")
ax.plot(data["predicted"][row], data["depths_m"], "s--", label="OceanEmbed")
ax.invert_yaxis(); ax.set(xlabel="Temperature (C)", ylabel="Depth (m)"); ax.legend(); ax.grid(alpha=.25)
st.pyplot(fig, use_container_width=False)
st.caption("Model output is a reconstruction, not an independent observation.")
