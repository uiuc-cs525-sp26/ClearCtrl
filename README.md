

We use RocksDB v10.10.1 (official signed release).

environment:
google cloud n2-standard-8 (8 vCPU, 32 GB DRAM)

apt install -y libgflags-dev

make -j2 static_lib PORTABLE=1 DISABLE_WARNING_AS_ERROR=1

make -j2 db_bench DEBUG_LEVEL=0 LIB_MODE=static DISABLE_WARNING_AS_ERROR=1

The user passes a `rocksdb::DB*` pointer to your controller module, and you launch a background thread to perform periodic monitoring and parameter tuning.
More precisely, however, you are monitoring RocksDB's internal state.

We provide a controller library. After creating and opening a RocksDB DB instance, the user passes the `DB*` pointer to the controller. The controller then launches a background thread to periodically read RocksDB's internal state and dynamically adjust compaction-related parameters via the runtime options API.

First, verify two things:

1. Which metrics can be read via the API?

2. Which parameters can be modified at runtime via the API?

These two factors determine whether your controller can operate as a closed-loop system.

In other words, your next step should be to:

Compile a table of "observable metrics / controllable knobs."

user thread:
```cpp
rocksdb::DB* db;
rocksdb::DB::Open(..., &db);
db->Put(...);
db->Get(...);

CompactionController ctl(db);
ctl.Start();
```

our thread:
```cpp
while (true) {
    auto stall = get_stall();
    auto backlog = get_l0_files();

    if (stall > threshold) {
        db->SetOptions({{"max_background_jobs", "4"}});
    } else {
        db->SetOptions({{"max_background_jobs", "2"}});
    }

    sleep(5);
}
```

build:

```sh
cmake -S . -B build
cmake --build build -j$(nproc)
```

run:

```sh
./build/clearctrl_test -h
```
