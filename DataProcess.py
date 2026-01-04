from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.plot import plotting_extent

DEM_PATH = Path("data/dem/dem30.tif")
RAINFALL_PATH = Path('data/climate/降水.xlsx')
LUCC_BASE_DIR = Path('data/lucc')
CMORPH_BASE_DIR = Path('data/cmorph-2021')
    

def load_rainfall_data():
    """Load rainfall data from Excel file."""
    try:
        return pd.read_excel(RAINFALL_PATH)
    except Exception as e:
        print(f"Warning: Could not load rainfall data: {e}")
        return None


def load_lucc(year: int = 2021):
    """Load Fenhe land use raster for a given year (ESRI Grid directory)."""
    lucc_dir = LUCC_BASE_DIR / f"fenhe_{year}"
    if not lucc_dir.exists():
        raise FileNotFoundError(f"LUCC directory not found: {lucc_dir}")

    dataset_path_candidates = [lucc_dir, lucc_dir / "w001001.adf", lucc_dir / "hdr.adf"]

    last_error = None
    for candidate in dataset_path_candidates:
        try:
            ds = rasterio.open(candidate)
            break
        except RasterioIOError as exc:
            last_error = exc
    else:
        raise RasterioIOError(
            f"Could not open LUCC dataset in {lucc_dir}. Last error: {last_error}")

    with ds:
        array = ds.read(1)
        transform = ds.transform
        crs = ds.crs
        profile = ds.profile

    return array, transform, crs, profile


def load_dem(dem_path: Path = DEM_PATH):
    """Load DEM raster band as float array with nodata masked as NaN."""
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")

    with rasterio.open(dem_path) as ds:
        band = ds.read(1).astype(np.float32)
        nodata = ds.nodata
        if nodata is not None:
            band = np.where(band == nodata, np.nan, band)
        transform = ds.transform
        crs = ds.crs
        profile = ds.profile

    return band, transform, crs, profile


def plot_dem(dem_array: np.ndarray, transform, crs):
    """Visualize DEM with terrain colormap and colorbar."""
    extent = plotting_extent(dem_array, transform)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(dem_array, cmap="terrain", extent=extent)
    ax.set_title("Digital Elevation Model")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label("Elevation")

    if crs:
        ax.text(0.01, 0.01, f"CRS: {crs.to_string()}", transform=ax.transAxes,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    fig.tight_layout()
    return fig, ax


def print_dem_summary(dem_array: np.ndarray, transform, crs, profile):
    pixel_width = transform.a
    pixel_height = abs(transform.e)
    nodata = profile.get("nodata")
    driver = profile.get("driver")

    print("\n" + "=" * 60)
    print("Digital Elevation Model Summary")
    print("=" * 60)
    print(f"Dimensions : {dem_array.shape[0]} rows × {dem_array.shape[1]} cols")
    print(f"Pixel size : {pixel_width:.2f} m × {pixel_height:.2f} m")
    print(f"NoData     : {nodata}")
    print(f"Driver     : {driver}")
    if crs:
        print("CRS        :", crs)
    print("Transform  :", transform)

    profile_copy = profile.copy()
    for key in ("transform", "crs"):
        profile_copy.pop(key, None)

    print("\nRaster profile (key fields):")
    pprint(profile_copy, sort_dicts=False, width=80)
    print("=" * 60 + "\n")


def print_lucc_summary(lucc_array: np.ndarray, transform, crs, profile, top_n: int = 10):
    nodata = profile.get("nodata")
    pixel_width = transform.a
    pixel_height = abs(transform.e)

    valid_mask = np.ones(lucc_array.shape, dtype=bool)
    if nodata is not None:
        valid_mask &= lucc_array != nodata

    valid_pixels = lucc_array[valid_mask]
    unique_vals, counts = np.unique(valid_pixels, return_counts=True) if valid_pixels.size else ([], [])
    sorted_stats = sorted(zip(unique_vals, counts), key=lambda x: x[1], reverse=True)

    total_valid = int(valid_pixels.size)
    print("\n" + "-" * 60)
    print("Fenhe Land Use Summary (2021)")
    print("-" * 60)
    print(f"Dimensions : {lucc_array.shape[0]} rows × {lucc_array.shape[1]} cols")
    print(f"Pixel size : {pixel_width:.2f} m × {pixel_height:.2f} m")
    print(f"NoData     : {nodata}")
    print(f"CRS        : {crs}")
    print(f"Transform  : {transform}")
    print(f"Valid pixels: {total_valid:,}")

    print("\nTop land use classes (value -> pixel count):")
    if not sorted_stats:
        print("  (no valid pixels)")
    else:
        for value, count in sorted_stats[:top_n]:
            percent = (count / total_valid * 100) if total_valid else 0.0
            print(f"  {value:>6}: {count:,} pixels ({percent:.2f}%)")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    # Load and display rainfall data
    # rainfall_data = load_rainfall_data()
    # if rainfall_data is not None:
    #     print("Rainfall data loaded successfully:")
    #     print(rainfall_data.head())

    # Load and display DEM data
    # dem, transform, crs, profile = load_dem()
    # print_dem_summary(dem, transform, crs, profile)
    # fig, ax = plot_dem(dem, transform, crs)
    # plt.show()

    # Load and display 2021 LUCC data
    lucc, lucc_transform, lucc_crs, lucc_profile = load_lucc(2021)
    print_lucc_summary(lucc, lucc_transform, lucc_crs, lucc_profile)