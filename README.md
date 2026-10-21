# OceanEmbed

**A Colab-first deep-learning project that reconstructs a subsurface temperature profile from sea-surface observations.**

OceanEmbed starts with a focused MVP: SST + sea-surface height (SSH) patches over the North Indian Ocean are encoded by a small CNN and decoded into temperature at selected depths. It is inspired by the SIH 2026 OceanEmbed problem statement, but is an independent project - not an SIH submission.

## Why this design

The model learns a compact `embedding` of the local surface ocean state. Its decoder predicts a vertical profile at the patch centre. Training on local patches makes the prediction location-aware, keeps the model small enough for free Google Colab, and supports a useful demo later.

```
SST patch + SSH patch -> CNN encoder -> 128-D embedding -> MLP -> temperature by depth
```

## MVP scope

- Region: 5-30 N, 45-105 E (North Indian Ocean)
- Inputs: OSTIA SST and DUACS SSH/SLA
- Target: GLORYS temperature profile
- Default depths: 0, 10, 20, 50, 100, 200, 300, and 500 m
- Experiments: SST-only versus SST+SSH
- Training: Google Colab (GPU optional; this model is intentionally modest)

SSS is deliberately an optional third channel, not a dependency for the first end-to-end result.

## Quick start in Google Colab

1. Clone this repository, then install dependencies:

   ```bash
   !git clone <YOUR_REPOSITORY_URL>
   %cd OceanEmbed
   !pip install -r requirements.txt
   ```

2. Place the three NetCDF inputs in `data/raw/` (or mount Drive and use its paths). Download data through the providers' official access routes and keep raw files out of Git.

3. Build local-patch training samples. Variable names can be adapted with the command-line options.

   ```bash
   !python -m src.oceanembed.data.prepare \
   --sst data/raw/ostia_sst.nc --ssh data/raw/duacs_ssh.nc --target data/raw/glorys_temp.nc \
   --output data/processed/oceanembed_mvp.npz --samples-per-day 20
   ```

4. Train SST-only and SST+SSH models:

   ```bash
   !python -m src.oceanembed.training.train --data data/processed/oceanembed_mvp.npz --channels 0 --run-name sst_only
   !python -m src.oceanembed.training.train --data data/processed/oceanembed_mvp.npz --channels 0 1 --run-name sst_ssh
   ```

5. Launch the demo after training:

   ```bash
   !streamlit run app/streamlit_app.py --server.port 8501
   ```

## Data contract

The preparation step expects gridded NetCDF data with a time dimension and latitude/longitude coordinates. It accepts common coordinate names (`latitude`/`lat`, `longitude`/`lon`) and the usual candidate variable names, but explicit options are available if a provider uses another name. It produces one sample per date/location:

```text
X: [N, channels, patch_height, patch_width]
y: [N, depths]
metadata: date, latitude, longitude, depth_m
```

Missing or land-filled windows are excluded. The scripts align every field to the SST daily grid before sampling.

## Evaluation

The trainer records overall MAE, RMSE, R2, Pearson correlation, and RMSE/MAE by depth. It saves:

- `results/<run>/best_model.pt`
- `results/<run>/metrics.json`
- `results/<run>/predictions.npz`
- predicted-versus-observed profile and depth-error plots

Use the two runs above to make a small, defensible ablation table rather than claiming a large-system benchmark.

## Project layout

```text
src/oceanembed/
  data/        # NetCDF alignment and patch preparation
  models/      # CNN encoder + profile decoder
  training/    # reproducible training loop
  evaluation/  # metrics and plots
app/           # Streamlit profile explorer
data/          # ignored raw/processed data
results/       # ignored models and figures
```

## Limitations and responsible claims

This is a research-style prototype, not an operational ocean product. Surface observations cannot fully determine all subsurface processes. Use chronological splits, report depth-wise errors, and reserve independent ARGO data for a future validation stage. Do not describe model outputs as observations.

## Next increments

1. Add SSS as a third channel.
2. Validate against gridded ARGO rather than only the GLORYS target.
3. Compare CNN embeddings with a lightweight transformer or temporal model.
