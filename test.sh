#! /bin/bash

./build/clearctrl_bench --log-path=logs/fixed2-500r.csv --controller=off --rounds=500 --rocksdb-max-background-jobs=2 --rocksdb-increase-parallelism=6
./build/clearctrl_bench --log-path=logs/fixed4-500r.csv --controller=off --rounds=500 --rocksdb-max-background-jobs=4 --rocksdb-increase-parallelism=6
./build/clearctrl_bench --log-path=logs/fixed6-500r.csv --controller=off --rounds=500 --rocksdb-max-background-jobs=6 --rocksdb-increase-parallelism=6
./build/clearctrl_bench --log-path=logs/controller-500r.csv --controller=on --rounds=500  