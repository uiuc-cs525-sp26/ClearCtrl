#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 仅服务当前论文 Motivation 图的固定配置。
CSV_PATH = Path("logs/motivation.csv")
RUN_CONFIG_PATH = Path("logs/motivation-run_config.json")
OUT_PNG = Path("analysis/motivation.png")
OUT_PDF = Path("analysis/motivation.pdf")
TAIL_WINDOW = 30


def infer_sleep_ms_from_schedule(schedule: str) -> float:
    # Expected format: OPS:SLEEP:ROUNDS;...
    # Motivation run is typically single phase; fallback to 0 if parsing fails.
    if not schedule:
        return 0.0
    first_phase = schedule.split(";")[0]
    parts = first_phase.split(":")
    if len(parts) != 3:
        return 0.0
    sleep_str = parts[1].strip()
    if not sleep_str:
        return 0.0
    try:
        return float(sleep_str)
    except ValueError:
        return 0.0


def main() -> None:
    # Black-and-white print friendly style:
    # distinguish curves with linestyle/width, not color only.
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
    })

    df_all = pd.read_csv(CSV_PATH)
    run_id_num = pd.to_numeric(df_all["run_id"], errors="coerce")
    if run_id_num.isna().all():
        raise SystemExit("run_id column cannot be parsed as numeric.")
    latest_run_id = int(run_id_num.dropna().iloc[-1])
    df = df_all[run_id_num == latest_run_id].copy().reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"run_id={latest_run_id} has no rows in {CSV_PATH}")

    with RUN_CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    l0_slowdown = cfg["rocksdb"]["l0_slowdown_trigger"]
    l0_stop = cfg["rocksdb"]["l0_stop_trigger"]
    sleep_ms = infer_sleep_ms_from_schedule(cfg.get("runner", {}).get("schedule", ""))

    # Use cumulative round duration as x-axis. CSV timestamp precision may be too coarse.
    latency_ms = pd.to_numeric(df["batch_latency_ms"], errors="coerce").fillna(0.0)
    round_ms = latency_ms + sleep_ms
    x = (round_ms.cumsum().to_numpy() - round_ms.iloc[0]) / 1000.0
    x_label = "Elapsed Time (s)"

    l0 = pd.to_numeric(df["l0_files"], errors="coerce")
    p95_tail = pd.to_numeric(df["batch_latency_ms"], errors="coerce").rolling(
        window=TAIL_WINDOW,
        min_periods=max(5, TAIL_WINDOW // 3),
    ).quantile(0.95)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)

    axes[0].plot(
        x, l0,
        color="black",
        linewidth=1.8,
        linestyle="-",
        label="L0 files",
    )
    axes[0].axhline(
        float(l0_slowdown),
        color="dimgray",
        linestyle="--",
        linewidth=1.4,
        label=f"slowdown={l0_slowdown}",
    )
    axes[0].axhline(
        float(l0_stop),
        color="gray",
        linestyle="-.",
        linewidth=1.4,
        label=f"stop={l0_stop}",
    )
    axes[0].set_ylabel("L0 Files")
    axes[0].set_title("(a) L0 File Count over Time")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        x,
        p95_tail,
        color="black",
        linewidth=1.8,
        linestyle=(0, (6, 2)),
        label=f"rolling p95 latency (w={TAIL_WINDOW})",
    )
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_xlabel(x_label)
    axes[1].set_title("(b) Tail Latency (P95) over Time")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.suptitle(
        "Motivation: Instability of Static Compaction under Sustained Write Pressure",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    print(f"[saved] {OUT_PNG}")
    print(f"[saved] {OUT_PDF}")
    print(f"[info] run_id={latest_run_id}, rows={len(df)}")


if __name__ == "__main__":
    main()
