"""Align NetCDF products and create valid local-patch samples.

Example:
python -m src.oceanembed.data.prepare --sst data/raw/ostia.nc --ssh data/raw/duacs.nc \
  --target data/raw/glorys.nc --output data/processed/mvp.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import xarray as xr

from src.oceanembed import DEPTHS_M

REGION = {"lat": slice(5, 30), "lon": slice(45, 105)}


def _rename_coords(ds: xr.Dataset) -> xr.Dataset:
    aliases = {"latitude": "lat", "longitude": "lon", "valid_time": "time"}
    return ds.rename({old: new for old, new in aliases.items() if old in ds.coords and new not in ds.coords})


def _variable(ds: xr.Dataset, requested: str | None, candidates: list[str]) -> xr.DataArray:
    name = requested or next((key for key in candidates if key in ds.data_vars), None)
    if name is None:
        raise KeyError(f"Could not find a variable. Available: {list(ds.data_vars)}. Use the corresponding --*-var option.")
    return ds[name]


def _open_field(path: str, requested: str | None, candidates: list[str]) -> xr.DataArray:
    ds = _rename_coords(xr.open_dataset(path))
    field = _variable(ds, requested, candidates).squeeze(drop=True)
    if "lon" not in field.coords or "lat" not in field.coords or "time" not in field.coords:
        raise ValueError(f"{path} must expose time, latitude, and longitude coordinates.")
    if float(field.lon.min()) < 0:
        field = field.assign_coords(lon=(field.lon % 360)).sortby("lon")
    # Several satellite products store latitude north-to-south; normalize the
    # coordinate order before slicing so the regional selection is reliable.
    return field.sortby("lat").sel(lat=REGION["lat"], lon=REGION["lon"]).sortby("time")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OceanEmbed patch samples from SST, SSH, and GLORYS NetCDF files.")
    parser.add_argument("--sst", required=True); parser.add_argument("--ssh", required=True); parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--sst-var"); parser.add_argument("--ssh-var"); parser.add_argument("--target-var")
    parser.add_argument("--depths", type=float, nargs="+", default=list(DEPTHS_M))
    parser.add_argument("--patch-size", type=int, default=16, help="Even width/height in SST-grid cells.")
    parser.add_argument("--samples-per-day", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.patch_size < 8 or args.patch_size % 2:
        raise ValueError("--patch-size must be an even integer of at least 8.")
    sst = _open_field(args.sst, args.sst_var, ["analysed_sst", "sst", "tos"])
    ssh = _open_field(args.ssh, args.ssh_var, ["adt", "sla", "ssh", "zos"])
    temp = _open_field(args.target, args.target_var, ["thetao", "temperature", "temp"])
    depth_name = next((d for d in ("depth", "deptht", "lev") if d in temp.coords), None)
    if depth_name is None:
        raise ValueError("The GLORYS target needs a depth coordinate (depth, deptht, or lev).")
    temp = temp.sel({depth_name: args.depths}, method="nearest").rename({depth_name: "depth"})
    # Daily SST is the reference grid/time. Interpolation makes spatial and temporal assumptions explicit.
    ssh = ssh.interp(time=sst.time, lat=sst.lat, lon=sst.lon)
    temp = temp.interp(time=sst.time, lat=sst.lat, lon=sst.lon)
    sst, ssh, temp = xr.align(sst, ssh, temp, join="inner")
    half = args.patch_size // 2
    rng = np.random.default_rng(args.seed)
    xs: list[np.ndarray] = []; ys: list[np.ndarray] = []; dates: list[str] = []; lats: list[float] = []; lons: list[float] = []
    for day in range(sst.sizes["time"]):
        # Materialize one day at a time: workable on free Colab and avoids loading whole archives.
        surface = np.stack([sst.isel(time=day).values, ssh.isel(time=day).values]).astype("float32")
        profile = temp.isel(time=day).transpose("depth", "lat", "lon").values.astype("float32")
        valid = np.isfinite(surface).all(axis=0) & np.isfinite(profile).all(axis=0)
        candidates = np.argwhere(valid)
        candidates = candidates[(candidates[:, 0] >= half) & (candidates[:, 0] < surface.shape[1] - half) & (candidates[:, 1] >= half) & (candidates[:, 1] < surface.shape[2] - half)]
        if len(candidates) == 0: continue
        for row, col in candidates[rng.choice(len(candidates), size=min(args.samples_per_day, len(candidates)), replace=False)]:
            patch = surface[:, row-half:row+half, col-half:col+half]
            if np.isfinite(patch).all():
                xs.append(patch); ys.append(profile[:, row, col]); dates.append(str(sst.time.values[day]))
                lats.append(float(sst.lat.values[row])); lons.append(float(sst.lon.values[col]))
    if not xs: raise RuntimeError("No valid samples were created. Check fields, region, and missing values.")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X=np.stack(xs), y=np.stack(ys), dates=np.asarray(dates), lat=np.asarray(lats), lon=np.asarray(lons), depths_m=np.asarray(temp.depth.values))
    print(f"Saved {len(xs)} samples: X={np.stack(xs).shape}, y={np.stack(ys).shape} -> {output}")


if __name__ == "__main__": main()
