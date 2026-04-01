#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "batch_latency_ms",
    "batch_payload_bytes",
    "stall_micros_total",
    "stall_micros_delta",
    "bg_jobs",
    "l0_files",
}


def infer_config_name(path: Path) -> str:
    stem = path.stem.lower()
    if stem.startswith("fixed2"):
        return "fixed2"
    if stem.startswith("fixed4"):
        return "fixed4"
    if stem.startswith("fixed6"):
        return "fixed6"
    if stem.startswith("controller"):
        return "controller"
    return stem


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少列: {sorted(missing)}")
    return df


def compute_overall_throughput_mb_s(df: pd.DataFrame) -> float:
    total_payload_bytes = float(df["batch_payload_bytes"].sum())
    total_latency_ms = float(df["batch_latency_ms"].sum())
    if total_latency_ms <= 0.0:
        return 0.0
    total_payload_mb = total_payload_bytes / (1024.0 * 1024.0)
    return total_payload_mb / (total_latency_ms / 1000.0)


def compute_summary(config: str, df: pd.DataFrame) -> Dict[str, float]:
    lat = df["batch_latency_ms"]
    overall_tp = compute_overall_throughput_mb_s(df)
    stall_total_s = float(df["stall_micros_total"].max()) / 1e6

    return {
        "config": config,
        "mean_latency_ms": float(lat.mean()),
        "p95_latency_ms": float(lat.quantile(0.95)),
        "p99_latency_ms": float(lat.quantile(0.99)),
        "max_latency_ms": float(lat.max()),
        "overall_throughput_mb_s": overall_tp,
        "total_stall_s": stall_total_s,
        "rows": int(len(df)),
    }


def controller_switch_analysis(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["bg_change"] = d["bg_jobs"].ne(d["bg_jobs"].shift()).fillna(True)
    switches = d[d["bg_change"]].copy()
    if switches.empty:
        return pd.DataFrame()
    switches["idx"] = switches.index
    return switches[
        [
            "idx",
            "bg_jobs",
            "l0_files",
            "stall_micros_total",
            "batch_payload_bytes",
            "batch_latency_ms",
        ]
    ]


def controller_by_bg_jobs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bg_jobs, group in df.groupby("bg_jobs", dropna=False):
        rows.append(
            {
                "bg_jobs": bg_jobs,
                "rounds": int(len(group)),
                "mean_latency_ms": float(group["batch_latency_ms"].mean()),
                "p95_latency_ms": float(group["batch_latency_ms"].quantile(0.95)),
                "p99_latency_ms": float(group["batch_latency_ms"].quantile(0.99)),
                "overall_throughput_mb_s": compute_overall_throughput_mb_s(group),
                "mean_l0": float(group["l0_files"].mean()),
                "stall_delta_sum_s": float(group["stall_micros_delta"].sum()) / 1e6,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("bg_jobs")


def compare_controller_vs_baselines(summary_df: pd.DataFrame) -> pd.DataFrame:
    if "controller" not in set(summary_df["config"]):
        return pd.DataFrame()

    ctrl = summary_df[summary_df["config"] == "controller"].iloc[0]
    baselines = summary_df[summary_df["config"] != "controller"].copy()
    if baselines.empty:
        return pd.DataFrame()

    metrics = [
        "mean_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "max_latency_ms",
        "overall_throughput_mb_s",
        "total_stall_s",
    ]

    rows = []
    for _, row in baselines.iterrows():
        out = {"baseline": row["config"]}
        for m in metrics:
            if row[m] == 0:
                out[m + "_delta_pct"] = np.nan
            else:
                out[m + "_delta_pct"] = (ctrl[m] - row[m]) / row[m] * 100.0
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 ClearCtrl logs/*.csv 指标")
    parser.add_argument(
        "csv_files",
        nargs="*",
        help="要分析的 CSV 文件；不传则默认读取 logs/*.csv",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="默认扫描目录（当不传 csv_files 时生效）",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis",
        help="输出目录（summary/switches 等结果）",
    )
    args = parser.parse_args()

    if args.csv_files:
        files = [Path(p) for p in args.csv_files]
    else:
        files = sorted(Path(args.logs_dir).glob("*.csv"))

    if not files:
        raise SystemExit("未找到可分析的 CSV 文件。")

    dfs: Dict[str, pd.DataFrame] = {}
    for p in files:
        config = infer_config_name(p)
        dfs[config] = load_csv(p)

    order = ["fixed2", "fixed4", "fixed6", "controller"]
    summaries = [compute_summary(cfg, dfs[cfg]) for cfg in order if cfg in dfs]
    for cfg in dfs:
        if cfg not in order:
            summaries.append(compute_summary(cfg, dfs[cfg]))

    summary_df = pd.DataFrame(summaries)
    if summary_df.empty:
        raise SystemExit("没有可用数据。")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"[saved] {summary_csv}")

    if "controller" in dfs:
        ctrl_df = dfs["controller"]
        switch_df = controller_switch_analysis(ctrl_df)
        by_bg_df = controller_by_bg_jobs(ctrl_df)
        cmp_df = compare_controller_vs_baselines(summary_df)

        if not switch_df.empty:
            switch_path = out_dir / "controller_switches.csv"
            switch_df.to_csv(switch_path, index=False)
            print(f"[saved] {switch_path}")
            print(f"[info] controller switch 次数: {len(switch_df)}")

        if not by_bg_df.empty:
            by_bg_path = out_dir / "controller_by_bg_jobs.csv"
            by_bg_df.to_csv(by_bg_path, index=False)
            print(f"[saved] {by_bg_path}")

        if not cmp_df.empty:
            cmp_path = out_dir / "controller_vs_baselines_pct.csv"
            cmp_df.to_csv(cmp_path, index=False)
            print(f"[saved] {cmp_path}")


if __name__ == "__main__":
    main()
