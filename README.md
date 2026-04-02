# ClearCtrl

ClearCtrl is a RocksDB compaction tuning prototype.
It runs a write workload and dynamically adjusts `max_background_jobs` at runtime.

## Environment

- RocksDB: `v10.10.1`
- Example machine: Google Cloud `n2-standard-8` (8 vCPU, 32 GB RAM)

## Build RocksDB (Static Library)

From `../rocksdb`:

```sh
apt install -y libgflags-dev
make -j2 static_lib PORTABLE=1 DISABLE_WARNING_AS_ERROR=1
```

Optional benchmark binary:

```sh
make -j2 db_bench DEBUG_LEVEL=0 LIB_MODE=static DISABLE_WARNING_AS_ERROR=1
```

## Build ClearCtrl

```sh
cmake -S . -B build
cmake --build build -j$(nproc)
```

Binary:

- `./build/clearctrl_bench`

## Run

Show all options:

```sh
./build/clearctrl_bench -h
```

Example fixed run:

```sh
./build/clearctrl_bench \
  --controller=off \
  --rounds=500 \
  --rocksdb-max-background-jobs=6 \
  --rocksdb-increase-parallelism=6 \
  --log-path=logs/fixed6.csv
```

Example phased run (low -> high -> low):

```sh
./build/clearctrl_bench \
  --controller=on \
  --runner-schedule="1000:100:500;4000:100:500;500:100:500" \
  --log-path=logs/controller.csv
```

`--runner-schedule` format:

- `OPS:SLEEP_MS:ROUNDS;OPS:SLEEP_MS:ROUNDS;...`

## Reproducible Test Workflow

Run all 4 workloads (fixed2/fixed4/fixed6/controller) and save console output:

```sh
bash ./test.sh 2>&1 | tee logs/test.log
```

Analyze logs:

```sh
python analyze_logs.py \
  logs/controller.csv \
  logs/fixed2.csv \
  logs/fixed4.csv \
  logs/fixed6.csv
```

Important:

- `analyze_logs.py` requires explicit input files.
- The first input file is treated as `controller/reference`.
- Remaining input files are treated as baselines.

Analysis outputs:

- `analysis/summary.csv`
- `analysis/controller_switches.csv`
- `analysis/controller_by_bg_jobs.csv`
- `analysis/controller_vs_baselines_pct.csv`

## Controller Logic (Current)

The controller is two-level (`low_bg_jobs` / `high_bg_jobs`) with cooldown.

- Scale up to high when any risk signal appears:
  - `actual_delayed_write_rate > 0`
  - `is_write_stopped > 0`
  - `stall_micros_delta > 0`
  - `l0_files > l0_slowdown_trigger`
- Scale down to low only after a sustained low-pressure window:
  - consecutive control cycles with:
    - `stall_micros_delta == 0`
    - `compaction_pending_bytes == 0`
    - `num_running_compactions == 0`
    - `actual_delayed_write_rate == 0`
    - `l0_files < l0_compaction_trigger`

## Current Logged Fields

Per-round CSV columns:

- `run_id`
- `phase_idx`
- `timestamp`
- `l0_files`
- `stall_micros_total`
- `stall_micros_delta`
- `compaction_pending_bytes`
- `is_write_stopped`
- `actual_delayed_write_rate`
- `num_running_compactions`
- `batch_payload_bytes`
- `batch_latency_ms`
- `bg_jobs`
- `process_cpu_user_sec`
- `process_cpu_sys_sec`
