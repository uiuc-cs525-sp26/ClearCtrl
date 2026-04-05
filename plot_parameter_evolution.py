#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CONTROLLER_CSV = Path("logs/controller.csv")
CONTROLLER_CONFIG = Path("logs/controller-run_config.json")
OUT_PNG = Path("analysis/parameter_evolution.png")
OUT_PDF = Path("analysis/parameter_evolution.pdf")


def load_latest_controller_run(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["run_id", "phase_idx", "batch_latency_ms", "bg_jobs"]:
        if col not in df.columns:
            raise ValueError(f"{path} missing column: {col}")

    run_id_num = pd.to_numeric(df["run_id"], errors="coerce")
    if run_id_num.isna().all():
        raise ValueError(f"{path} run_id cannot be parsed")
    latest_run_id = int(run_id_num.dropna().iloc[-1])
    out = df[run_id_num == latest_run_id].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{path} latest run_id={latest_run_id} has no rows")
    return out


def load_phase_sleep_ms(config_path: Path) -> dict[int, float]:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    runner = cfg.get("runner", {})
    schedule = str(runner.get("schedule", "")).strip()
    default_sleep = float(runner.get("default_sleep_ms", 0.0))
    if not schedule:
        return {}

    phase_sleep: dict[int, float] = {}
    parts = [seg.strip() for seg in schedule.split(";") if seg.strip()]
    for i, seg in enumerate(parts, start=1):
        tokens = seg.split(":")
        if len(tokens) != 3:
            continue
        sleep_token = tokens[1].strip()
        if sleep_token == "":
            phase_sleep[i] = default_sleep
            continue
        try:
            phase_sleep[i] = float(sleep_token)
        except ValueError:
            phase_sleep[i] = default_sleep
    return phase_sleep


def add_elapsed_time(df: pd.DataFrame, phase_sleep_ms: dict[int, float]) -> pd.DataFrame:
    phase_idx = pd.to_numeric(df["phase_idx"], errors="coerce").fillna(0).astype(int)
    sleep_ms_series = phase_idx.map(phase_sleep_ms).fillna(0.0)
    latency_ms = pd.to_numeric(df["batch_latency_ms"], errors="coerce").fillna(0.0)
    round_ms = latency_ms + sleep_ms_series
    elapsed_s = (round_ms.cumsum() - round_ms.iloc[0]) / 1000.0

    out = df.copy()
    out["elapsed_s"] = elapsed_s
    out["bg_jobs"] = pd.to_numeric(out["bg_jobs"], errors="coerce")
    out["phase_idx"] = phase_idx
    return out


def phase_spans(df: pd.DataFrame) -> list[tuple[int, float, float]]:
    spans: list[tuple[int, float, float]] = []
    for p in sorted(df["phase_idx"].dropna().unique()):
        mask = df["phase_idx"] == p
        if not mask.any():
            continue
        s = float(df.loc[mask, "elapsed_s"].iloc[0])
        e = float(df.loc[mask, "elapsed_s"].iloc[-1])
        spans.append((int(p), s, e))
    return spans


def setup_bw_style() -> None:
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
    })


def main() -> None:
    setup_bw_style()

    df = load_latest_controller_run(CONTROLLER_CSV)
    phase_sleep_ms = load_phase_sleep_ms(CONTROLLER_CONFIG)
    d = add_elapsed_time(df, phase_sleep_ms)
    spans = phase_spans(d)

    fig, ax = plt.subplots(figsize=(11, 4.2))

    # Highlight phase 2 (and other even phases) with light gray bands.
    for phase_id, start_s, end_s in spans:
        if phase_id % 2 == 0:
            ax.axvspan(start_s, end_s, facecolor="lightgray", alpha=0.25, zorder=0)

    ax.step(
        d["elapsed_s"],
        d["bg_jobs"],
        where="post",
        color="black",
        linewidth=2.4,
        linestyle="-",
        label="ClearCtrl max_background_jobs",
    )

    ax.set_title("Parameter Evolution: max_background_jobs over Time")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("max_background_jobs")
    ax.set_yticks(sorted(d["bg_jobs"].dropna().unique()))
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    print(f"[saved] {OUT_PNG}")
    print(f"[saved] {OUT_PDF}")


if __name__ == "__main__":
    main()
