# ClearCtrl

ClearCtrl is a RocksDB compaction tuning prototype.

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
  --log-path=logs/fixed6-500r.csv
```

## Reproducible Test Workflow

Run all 4 workloads (fixed2/fixed4/fixed6/controller) and save console output:

```sh
sh ./test.sh 2>&1 | tee logs/test.log
```

Analyze logs:

```sh
python analyze_logs.py \
  logs/fixed2-500r.csv \
  logs/fixed4-500r.csv \
  logs/fixed6-500r.csv \
  logs/controller-500r.csv
```

Analysis outputs:

- `analysis/summary.csv`
- `analysis/controller_switches.csv`
- `analysis/controller_by_bg_jobs.csv`
- `analysis/controller_vs_baselines_pct.csv`

## Current Logged Fields

Per-round CSV columns:

- `run_id`
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
