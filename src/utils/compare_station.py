"""Compare station observations with CMORPH daily aggregations.

This script computes per-station error metrics between ground observations and
aggregated satellite precipitation under two accumulation systems:
- Hydrological (08:00-08:00)
- Meteorological (20:00-20:00)

Usage example:

python scripts/compare_station_alignment.py \
    --hydro-csv data/cmorph-2021/daily/fenhe_hydro_08-08_2021.csv \
    --cma-csv data/cmorph-2021/daily/fenhe_cma_20-20_2021.csv \
    --meta-path data/climate/meta.xlsx \
    --rain-path data/climate/rain.xlsx \
    --year 2021 \
    --output-csv output/station_alignment_summary.csv \
    --plot-stations 53465 53512 \
    --plot-dir output/station_plots
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_satellite_table(csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load aggregated precipitation CSV into a pivoted table.

    Returns a DataFrame indexed by time with MultiIndex columns (lat, lon),
    along with sorted unique latitude and longitude arrays for nearest-neighbour queries.
    """
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.dropna(subset=["precip"]).copy()

    if df.empty:
        raise ValueError(f"{csv_path} 不包含有效降水记录")

    lat_values = np.sort(df["lat"].unique())
    lon_values = np.sort(df["lon"].unique())

    pivot = df.pivot_table(index="time", columns=["lat", "lon"], values="precip")
    pivot = pivot.sort_index()

    return pivot, lat_values, lon_values


def select_nearest_coordinate(
    station_lat: float,
    station_lon: float,
    lat_values: np.ndarray,
    lon_values: np.ndarray,
) -> Tuple[float, float]:
    """Find the nearest latitude/longitude on the CMORPH grid."""
    lat_idx = lat_values[np.argmin(np.abs(lat_values - station_lat))]
    lon_idx = lon_values[np.argmin(np.abs(lon_values - station_lon))]
    return float(lat_idx), float(lon_idx)


def compute_metrics(obs: pd.Series, sat: pd.Series) -> dict:
    """Compute MAE, RMSE, correlation and overlap length between two series."""
    df = pd.concat({"obs": obs, "sat": sat}, axis=1, join="inner").dropna()
    if df.empty:
        return {"mae": math.nan, "rmse": math.nan, "corr": math.nan, "n": 0}

    diff = df["sat"] - df["obs"]
    mae = diff.abs().mean()
    rmse = math.sqrt((diff**2).mean())
    corr = df["sat"].corr(df["obs"])
    return {"mae": mae, "rmse": rmse, "corr": corr, "n": len(df)}


def plot_station_timeseries(
    station_id: str,
    obs: pd.Series,
    hydro: pd.Series,
    cma: pd.Series,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(obs.index, obs.values, label="Observation", color="#2E7D32")
    ax.plot(hydro.index, hydro.values, label="Hydro 08-08", color="#1565C0", alpha=0.8)
    ax.plot(cma.index, cma.values, label="CMA 20-20", color="#EF6C00", alpha=0.8)
    ax.set_title(f"Station {station_id} Comparison")
    ax.set_ylabel("Precipitation (mm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"station_{station_id}_comparison.png", dpi=200)
    plt.close(fig)


def prepare_station_series(meta_path: Path, rain_path: Path, year: int) -> pd.DataFrame:
    df_meta = pd.read_excel(meta_path, usecols=["F_站号", "经度", "纬度"])
    df_rain = (
        pd.read_excel(rain_path)
        .query("year == @year")
        .sort_values(["year", "month", "day"])
        .reset_index(drop=True)
    )

    if df_rain.empty:
        raise ValueError(f"{rain_path} 中没有 {year} 年的数据")

    df_rain["date"] = pd.to_datetime(df_rain[["year", "month", "day"]])
    df_rain = df_rain.drop(columns=["year", "month", "day"], errors="ignore")
    df_rain = df_rain.set_index("date")

    # 合并坐标，便于遍历
    df_meta["station_id"] = df_meta["F_站号"].astype(int).astype(str)
    return df_meta, df_rain


def compare_systems(
    hydro_csv: Path,
    cma_csv: Path,
    meta_path: Path,
    rain_path: Path,
    year: int,
    station_ids: Optional[Iterable[str]] = None,
    plot_ids: Optional[Iterable[str]] = None,
    plot_dir: Optional[Path] = None,
) -> pd.DataFrame:
    hydro_df, hydro_lats, hydro_lons = load_satellite_table(hydro_csv)
    cma_df, cma_lats, cma_lons = load_satellite_table(cma_csv)

    if not np.array_equal(hydro_lats, cma_lats) or not np.array_equal(hydro_lons, cma_lons):
        raise ValueError("Hydro 与 CMA 数据网格不一致，请检查输入文件。")

    meta_df, rain_df = prepare_station_series(meta_path, rain_path, year)

    target_ids = (
        set(station_ids)
        if station_ids is not None
        else set(meta_df["station_id"].tolist())
    )
    if plot_ids is not None:
        target_ids.update(plot_ids)

    results = []

    for _, row in meta_df.iterrows():
        station_id = row["station_id"]
        if station_id not in target_ids:
            continue

        if station_id not in rain_df.columns:
            print(f"站点 {station_id} 在雨量表中缺失，跳过")
            continue

        obs_series = rain_df[station_id]
        lat_idx, lon_idx = select_nearest_coordinate(
            row["纬度"], row["经度"], hydro_lats, hydro_lons
        )

        column_key = (lat_idx, lon_idx)
        if column_key not in hydro_df.columns:
            print(f"站点 {station_id} 对应网格 ({lat_idx}, {lon_idx}) 缺失于 Hydro 数据")
            continue
        if column_key not in cma_df.columns:
            print(f"站点 {station_id} 对应网格 ({lat_idx}, {lon_idx}) 缺失于 CMA 数据")
            continue

        hydro_series = hydro_df[column_key]
        cma_series = cma_df[column_key]

        hydro_metrics = compute_metrics(obs_series, hydro_series)
        cma_metrics = compute_metrics(obs_series, cma_series)

        better = None
        if not math.isnan(hydro_metrics["rmse"]) and not math.isnan(cma_metrics["rmse"]):
            better = "hydro" if hydro_metrics["rmse"] <= cma_metrics["rmse"] else "cma"

        results.append(
            {
                "station_id": station_id,
                "lat": row["纬度"],
                "lon": row["经度"],
                "hydro_mae": hydro_metrics["mae"],
                "hydro_rmse": hydro_metrics["rmse"],
                "hydro_corr": hydro_metrics["corr"],
                "hydro_overlap_days": hydro_metrics["n"],
                "cma_mae": cma_metrics["mae"],
                "cma_rmse": cma_metrics["rmse"],
                "cma_corr": cma_metrics["corr"],
                "cma_overlap_days": cma_metrics["n"],
                "better_system": better,
            }
        )

        if plot_ids and station_id in plot_ids and plot_dir is not None:
            plot_station_timeseries(
                station_id,
                obs_series,
                hydro_series,
                cma_series,
                plot_dir,
            )

    if not results:
        raise ValueError("未生成任何对比结果，请检查输入站点是否匹配。")

    return pd.DataFrame(results).sort_values("station_id")


def main():
    parser = argparse.ArgumentParser(description="Compare station rainfall with CMORPH systems")
    parser.add_argument("--hydro-csv", type=Path, required=True, help="Hydrological aggregation CSV 文件路径")
    parser.add_argument("--cma-csv", type=Path, required=True, help="Meteorological aggregation CSV 文件路径")
    parser.add_argument("--meta-path", type=Path, required=True, help="雨量站元数据 (meta.xlsx)")
    parser.add_argument("--rain-path", type=Path, required=True, help="雨量站逐日数据 (rain.xlsx)")
    parser.add_argument("--year", type=int, default=2021, help="对齐年份 (默认: 2021)")
    parser.add_argument("--station-ids", nargs="*", help="可选：仅比较指定站点 ID")
    parser.add_argument("--plot-stations", nargs="*", help="可选：输出对比图的站点 ID")
    parser.add_argument("--plot-dir", type=Path, help="保存对比图的目录")
    parser.add_argument("--output-csv", type=Path, help="保存结果摘要的 CSV 路径")

    args = parser.parse_args()

    summary = compare_systems(
        hydro_csv=args.hydro_csv,
        cma_csv=args.cma_csv,
        meta_path=args.meta_path,
        rain_path=args.rain_path,
        year=args.year,
        station_ids=args.station_ids,
        plot_ids=args.plot_stations,
        plot_dir=args.plot_dir,
    )

    print("=== 对齐结果概览 ===")
    print(summary[[
        "station_id",
        "hydro_rmse",
        "cma_rmse",
        "hydro_corr",
        "cma_corr",
        "better_system"
    ]].to_string(index=False))

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_csv, index=False)
        print(f"结果已保存至 {args.output_csv}")


if __name__ == "__main__":
    main()
