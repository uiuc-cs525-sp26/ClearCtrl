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

Example fixed run (controller off, pin to fast-arm index 4 = `6:4`):

```sh
./build/clearctrl_bench \
  --controller=off \
  --runner-schedule="1000:100:500" \
  --ctrl-bandit-initial-arm=4 \
  --ctrl-bandit-initial-profile=2 \
  --log-path=logs/fixed6.csv
```

Example phased run (low -> high -> low) with controller on:

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

## Controller (LinUCB Contextual Bandit)

The controller uses two LinUCB models running at different cadences:

- **Fast bandit** (every `--ctrl-interval-sec`, gated by `--ctrl-cooldown-sec`)
  picks an arm `(max_background_jobs, max_subcompactions)` and applies it
  via `SetDBOptions`.
  - Action set from `--ctrl-bandit-arms="BG:SUB;BG:SUB;..."`
    (default `2:1;4:1;4:2;6:2;6:4`).
- **Slow profile bandit** (every `--ctrl-bandit-profile-cycles` fast cycles)
  picks an arm `(level0_slowdown_writes_trigger, level0_stop_writes_trigger)`
  and applies it via `SetOptions` on the default column family.
  - Action set from `--ctrl-bandit-profiles="SLOW:STOP;SLOW:STOP;..."`
    (default `8:12;12:16;16:24`).
  - Each profile must satisfy **`level0_stop_writes_trigger` ≥
    `level0_slowdown_writes_trigger` ≥
    `level0_file_num_compaction_trigger`** (the compaction trigger is
    `--rocksdb-l0-compaction-trigger`, default `8`). The binary rejects invalid
    combinations at startup.

Initial state for both `DB::Open` and the bandit comes from two indices into
the action sets above:

- `--ctrl-bandit-initial-arm=N` (default `0`) picks the startup
  `max_background_jobs` / `max_subcompactions`.
- `--ctrl-bandit-initial-profile=N` (default `0`) picks the startup
  `level0_slowdown_writes_trigger` / `level0_stop_writes_trigger`.

These two indices are also used when `--controller=off` to pin the static
RocksDB configuration, so baseline runs no longer take separate
`--rocksdb-max-background-jobs` / `--rocksdb-max-subcompactions` /
`--rocksdb-l0-slowdown-trigger` / `--rocksdb-l0-stop-trigger` flags.

Per-arm linear model (LinUCB):

- For each arm `a`: maintain `A_a = lambda*I + sum x x^T`, `b_a = sum r * x`.
  We update `A_a^{-1}` incrementally with Sherman-Morrison and never solve a
  linear system at decision time.
- Pick `argmax_a (theta_a^T x + alpha * sqrt(x^T A_a^{-1} x))` where
  `theta_a = A_a^{-1} b_a` and `alpha` is `--ctrl-bandit-alpha`.

8-dim context vector (`x`) — **fast bandit, per-cycle instantaneous**:

1. bias = 1
2. `l0_files / l0_base`                           (clip [0,2])
3. `compaction_pending_bytes / 64MiB`             (clip [0,2])
4. `stall_micros_delta / interval_micros`         (clip [0,1])
5. `is_write_stopped`                             (0/1)
6. `num_running_compactions / current_bg_jobs`    (clip [0,2])
7. `recent_throughput_MB_s / tp_norm`             (clip [0,2])
8. `actual_delayed_write_rate > 0`                (0/1)

8-dim context vector — **profile bandit, window-aggregated state summary**.
Built once at the end of each profile window from accumulators across all
`profile_interval_cycles` fast cycles. Same shape as the fast context, but
all components are window aggregates:

1. bias = 1
2. `max(l0_files in window) / l0_base`            (clip [0,2])
3. `avg(compaction_pending_bytes) / 64MiB`        (clip [0,2])
4. `sum(stall_micros) / sum(interval_micros)`     (clip [0,1])
5. `1` iff `is_write_stopped` was ever observed in the window, else `0`
6. `avg(num_running_compactions / current_bg_jobs)` (clip [0,2])
7. `window_throughput_MB_s / tp_norm`             (clip [0,2])
8. `delayed_active_fraction` (cycles in window with `delayed_rate > 0` /
   total cycles in window)                        (clip [0,1])

**Profile bandit time-ordering.** To avoid leaking post-action observations
into the decision context, Update and SelectArm use *different* contexts at
each window boundary, but both come from this same window-aggregated
template:

```
t_window_start :  (we already committed arm a_t earlier, with decision
                   context x_t snapshotted at that moment)
... window runs, accumulators fill ...
t_window_end   :  observe window outcome, compute profile_reward r_t
                  Update(a_t, x_t, r_t)            <-- pre-action context
                  Build profile_state_x for the just-ended window
                  a_{t+1} = SelectArm(profile_state_x)
                  pending_decision_x for a_{t+1}  := profile_state_x
                  apply a_{t+1}
```

The constructor-set initial profile is NOT a bandit decision and therefore
receives NO Update at the first window close (we have no decision context
for it).

Fast-bandit reward (per fast cycle, computed from the window since the last
decision):

```
fast_reward = (recent_throughput_MB_s / tp_norm)
            - lambda_stall * stall_fraction
            - lambda_l0    * (max_l0 / l0_base)
```

Profile-bandit reward (computed once per profile window, aggregated over the
entire `--ctrl-bandit-profile-cycles` worth of fast cycles, then used to do a
single LinUCB update for the profile actually in effect during that window):

```
profile_reward = fast_reward(window-aggregated metrics)
               - lambda_stop  * stop_seen_in_window      (0 or 1)
               - lambda_delay * delayed_active_fraction  (in [0,1])
```

- `stop_seen_in_window` is `1` if `is_write_stopped` was observed at any
  cycle inside the window, else `0`.
- `delayed_active_fraction` is the fraction of cycles in the window where
  `actual_delayed_write_rate > 0`.

These two extra penalties exist only on the profile reward because the
profile bandit is what controls `level0_slowdown_writes_trigger` /
`level0_stop_writes_trigger`. Without them, raising those triggers can defer
backlog signals long enough to look "fine" on tp/stall/l0 alone while the
system is already in delayed/stopped territory.

`l0_base` is the FIXED `level0_file_num_compaction_trigger`
(`--rocksdb-l0-compaction-trigger`), not the profile-controlled
`level0_slowdown_writes_trigger`. Using a fixed base prevents the profile
bandit from trivially shrinking its own L0 penalty by raising the slowdown
trigger when actual backlog has not improved.

Tunable via `--ctrl-bandit-reward-tp-norm-mb-s`,
`--ctrl-bandit-reward-lambda-stall`, `--ctrl-bandit-reward-lambda-l0`,
`--ctrl-bandit-reward-lambda-stop`, `--ctrl-bandit-reward-lambda-delay`,
`--rocksdb-l0-compaction-trigger`.

Throughput is read from RocksDB's `BYTES_WRITTEN` ticker delta, so it reflects
end-to-end write bandwidth between two controller cycles, independent of the
foreground batch loop.

Example controller run:

```sh
./build/clearctrl_bench \
  --controller=on \
  --runner-schedule="1000:100:300;4000:100:300;500:100:300" \
  --ctrl-bandit-arms="2:1;4:1;4:2;6:2;6:4" \
  --ctrl-bandit-profiles="8:12;12:16;16:24" \
  --ctrl-bandit-initial-arm=0 \
  --ctrl-bandit-initial-profile=0 \
  --ctrl-bandit-alpha=1.0 \
  --ctrl-bandit-profile-cycles=10 \
  --log-path=logs/controller.csv
```

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
- `subcompactions`           (current `max_subcompactions`)
- `l0_slowdown_now`          (current `level0_slowdown_writes_trigger`)
- `l0_stop_now`              (current `level0_stop_writes_trigger`)
- `arm_id`                   (LinUCB fast-arm id; `-1` if controller off)
- `profile_id`               (LinUCB profile-arm id; `-1` if controller off)
- `last_fast_reward`         (most recent fast-bandit reward, per-cycle; `0` if controller off)
- `last_profile_reward`      (most recent profile-bandit reward, includes
                              stop / delay penalties, only refreshed at
                              profile-window boundaries; `0` if controller off)
