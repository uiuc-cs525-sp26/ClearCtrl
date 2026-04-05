#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Fixed inputs for current project workflow.
RUNS = [
    ("controller", Path("logs/controller.csv")),
    ("fixed2", Path("logs/fixed2.csv")),
    ("fixed4", Path("logs/fixed4.csv")),
    ("fixed6", Path("logs/fixed6.csv")),
]

OUT_PNG = Path("analysis/l0_vs_time.png")
OUT_PDF = Path("analysis/l0_vs_time.pdf")

# Keep consistent with bench_main defaults for this figure.
L0_COMPACTION_TRIGGER = 8
L0_SLOWDOWN_TRIGGER = 12
L0_SMOOTH_WINDOW = 10


def load_latest_run(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["run_id", "phase_idx", "batch_latency_ms", "l0_files"]:
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


def load_phase_sleep_ms(csv_path: Path) -> dict[int, float]:
    # Per-run config is expected at: logs/<stem>-run_config.json
    cfg_path = csv_path.with_name(f"{csv_path.stem}-run_config.json")
    if not cfg_path.exists():
        return {}

    with cfg_path.open("r", encoding="utf-8") as f:
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


def add_metrics(df: pd.DataFrame, phase_sleep_ms: dict[int, float]) -> pd.DataFrame:
    # Reconstruct benchmark-aligned elapsed time:
    # round_duration = batch_latency_ms + sleep_ms_of_current_phase
    phase_idx = pd.to_numeric(df["phase_idx"], errors="coerce").fillna(0).astype(int)
    sleep_ms_series = phase_idx.map(phase_sleep_ms).fillna(0.0)

    latency_ms = pd.to_numeric(df["batch_latency_ms"], errors="coerce")
    round_ms = latency_ms.fillna(0.0) + sleep_ms_series
    elapsed = (round_ms.cumsum() - round_ms.iloc[0]) / 1000.0

    out = df.copy()
    out["elapsed_s"] = elapsed
    out["l0_smooth"] = pd.to_numeric(out["l0_files"], errors="coerce").rolling(
        window=L0_SMOOTH_WINDOW,
        min_periods=max(3, L0_SMOOTH_WINDOW // 3),
    ).mean()
    return out


def phase_boundary_times(df: pd.DataFrame) -> list[float]:
    phase = pd.to_numeric(df["phase_idx"], errors="coerce")
    change_mask = phase.ne(phase.shift(1))
    idx = df.index[change_mask].tolist()
    if idx:
        idx = idx[1:]
    return [float(df.loc[i, "elapsed_s"]) for i in idx]


def phase_spans(df: pd.DataFrame) -> list[tuple[int, float, float]]:
    phase = pd.to_numeric(df["phase_idx"], errors="coerce").fillna(0).astype(int)
    elapsed = pd.to_numeric(df["elapsed_s"], errors="coerce")
    spans: list[tuple[int, float, float]] = []
    for p in sorted(phase.unique()):
        mask = phase == p
        if not mask.any():
            continue
        s = float(elapsed[mask].iloc[0])
        e = float(elapsed[mask].iloc[-1])
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

    datasets: dict[str, pd.DataFrame] = {}
    for label, path in RUNS:
        phase_sleep_ms = load_phase_sleep_ms(path)
        datasets[label] = add_metrics(load_latest_run(path), phase_sleep_ms)

    boundaries = phase_boundary_times(datasets["controller"])
    spans = phase_spans(datasets["controller"])

    # Black-and-white friendly line styles.
    styles = {
        "controller": dict(color="black", linestyle="-", linewidth=2.6),
        "fixed2": dict(color="dimgray", linestyle="--", linewidth=1.3),
        "fixed4": dict(color="gray", linestyle="-.", linewidth=1.3),
        "fixed6": dict(color="black", linestyle=(0, (2, 2)), linewidth=1.3),
    }
    legend_names = {
        "controller": "ClearCtrl",
        "fixed2": "Fixed-2",
        "fixed4": "Fixed-4",
        "fixed6": "Fixed-6",
    }

    fig, ax = plt.subplots(figsize=(11, 4.6))

    # Phase background bands: highlight high-pressure middle phase.
    for phase_id, start_s, end_s in spans:
        if phase_id % 2 == 0:
            ax.axvspan(start_s, end_s, facecolor="lightgray", alpha=0.25, zorder=0)

    for label, df in datasets.items():
        ax.plot(
            df["elapsed_s"],
            df["l0_smooth"],
            label=legend_names.get(label, label),
            **styles[label],
        )

    for x in boundaries:
        ax.axvline(x, color="silver", linestyle=":", linewidth=0.9)

    ax.axhline(
        L0_COMPACTION_TRIGGER,
        color="black",
        linestyle=(0, (6, 2)),
        linewidth=1.2,
        label=f"compaction_trigger={L0_COMPACTION_TRIGGER}",
    )
    ax.axhline(
        L0_SLOWDOWN_TRIGGER,
        color="black",
        linestyle=(0, (1, 2)),
        linewidth=1.2,
        label=f"slowdown_trigger={L0_SLOWDOWN_TRIGGER}",
    )

    ax.set_title("L0 Files vs Time")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("L0 Files, rolling mean")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncol=2, frameon=False)
    fig.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    print(f"[saved] {OUT_PNG}")
    print(f"[saved] {OUT_PDF}")


if __name__ == "__main__":
    main()
