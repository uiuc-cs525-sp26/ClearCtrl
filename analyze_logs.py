#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "phase_idx",
    "batch_latency_ms",
    "batch_payload_bytes",
    "stall_micros_total",
    "stall_micros_delta",
    "bg_jobs",
    "l0_files",
    "process_cpu_user_sec",
    "process_cpu_sys_sec",
}

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


def compute_cpu_totals(df: pd.DataFrame) -> Dict[str, float]:
    user_total = float(df["process_cpu_user_sec"].max() - df["process_cpu_user_sec"].min())
    sys_total = float(df["process_cpu_sys_sec"].max() - df["process_cpu_sys_sec"].min())
    total = user_total + sys_total
    return {
        "cpu_user_sec": user_total,
        "cpu_sys_sec": sys_total,
        "cpu_total_sec": total,
    }


def compute_summary(file_label: str, df: pd.DataFrame) -> Dict[str, float]:
    lat = df["batch_latency_ms"]
    overall_tp = compute_overall_throughput_mb_s(df)
    stall_total_s = float(df["stall_micros_total"].max()) / 1e6
    cpu = compute_cpu_totals(df)
    total_payload_mb = float(df["batch_payload_bytes"].sum()) / (1024.0 * 1024.0)
    cpu_sec_per_mb = np.nan
    if total_payload_mb > 0.0:
        cpu_sec_per_mb = cpu["cpu_total_sec"] / total_payload_mb

    return {
        "file": file_label,
        "mean_latency_ms": float(lat.mean()),
        "p95_latency_ms": float(lat.quantile(0.95)),
        "p99_latency_ms": float(lat.quantile(0.99)),
        "max_latency_ms": float(lat.max()),
        "overall_throughput_mb_s": overall_tp,
        "total_stall_s": stall_total_s,
        "process_cpu_user_sec": cpu["cpu_user_sec"],
        "process_cpu_sys_sec": cpu["cpu_sys_sec"],
        "process_cpu_total_sec": cpu["cpu_total_sec"],
        "cpu_sec_per_mb": cpu_sec_per_mb,
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
            "phase_idx",
            "bg_jobs",
            "l0_files",
            "stall_micros_total",
            "batch_payload_bytes",
            "batch_latency_ms",
            "process_cpu_user_sec",
            "process_cpu_sys_sec",
        ]
    ]


def controller_by_bg_jobs(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["cpu_total_delta_sec"] = (
        d["process_cpu_user_sec"].diff().fillna(0.0) +
        d["process_cpu_sys_sec"].diff().fillna(0.0)
    ).clip(lower=0.0)
    rows = []
    for bg_jobs, group in d.groupby("bg_jobs", dropna=False):
        payload_mb = float(group["batch_payload_bytes"].sum()) / (1024.0 * 1024.0)
        cpu_total_delta_sec = float(group["cpu_total_delta_sec"].sum())
        cpu_sec_per_mb = np.nan
        if payload_mb > 0.0:
            cpu_sec_per_mb = cpu_total_delta_sec / payload_mb
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
                "cpu_total_delta_sec": cpu_total_delta_sec,
                "cpu_sec_per_mb": cpu_sec_per_mb,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("bg_jobs")


def compare_reference_vs_baselines(summary_df: pd.DataFrame, ref_file: str) -> pd.DataFrame:
    if ref_file not in set(summary_df["file"]):
        return pd.DataFrame()

    ref = summary_df[summary_df["file"] == ref_file].iloc[0]
    baselines = summary_df[summary_df["file"] != ref_file].copy()
    if baselines.empty:
        return pd.DataFrame()

    metrics = [
        "mean_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "max_latency_ms",
        "overall_throughput_mb_s",
        "total_stall_s",
        "process_cpu_total_sec",
        "cpu_sec_per_mb",
    ]

    rows = []
    for _, row in baselines.iterrows():
        out = {"baseline_file": row["file"], "reference_file": ref_file}
        for m in metrics:
            if row[m] == 0:
                out[m + "_delta_pct"] = np.nan
            else:
                out[m + "_delta_pct"] = (ref[m] - row[m]) / row[m] * 100.0
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 ClearCtrl logs/*.csv 指标")
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="要分析的 CSV 文件（必须显式传入；顺序约定：第一个是 controller/reference，其余是 baselines）",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis",
        help="输出目录（summary/switches 等结果）",
    )
    args = parser.parse_args()

    files = [Path(p) for p in args.csv_files]

    if not files:
        raise SystemExit("未找到可分析的 CSV 文件。")

    datasets: List[Tuple[str, pd.DataFrame]] = []
    name_count: Dict[str, int] = {}
    for p in files:
        label = p.name
        count = name_count.get(label, 0)
        name_count[label] = count + 1
        if count > 0:
            label = f"{label}#{count+1}"
        datasets.append((label, load_csv(p)))

    summaries = [compute_summary(label, df) for label, df in datasets]

    summary_df = pd.DataFrame(summaries)
    if summary_df.empty:
        raise SystemExit("没有可用数据。")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"[saved] {summary_csv}")
    print(f"[info] reference(controller) file: {datasets[0][0]}")

    # By convention: first file is controller/reference, rest are baselines.
    if datasets:
        ctrl_label, ctrl_df = datasets[0]
        switch_df = controller_switch_analysis(ctrl_df)
        by_bg_df = controller_by_bg_jobs(ctrl_df)
        cmp_df = compare_reference_vs_baselines(summary_df, ctrl_label)

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
