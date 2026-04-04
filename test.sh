#! /bin/bash
set -euo pipefail

./build/clearctrl_bench --log-path=logs/fixed2.csv --controller=off --runner-schedule="1000:100:300;4000:100:300;500:100:300" --rocksdb-max-background-jobs=2 --rocksdb-increase-parallelism=6
./build/clearctrl_bench --log-path=logs/fixed4.csv --controller=off --runner-schedule="1000:100:300;4000:100:300;500:100:300" --rocksdb-max-background-jobs=4 --rocksdb-increase-parallelism=6
./build/clearctrl_bench --log-path=logs/fixed6.csv --controller=off --runner-schedule="1000:100:300;4000:100:300;500:100:300" --rocksdb-max-background-jobs=6 --rocksdb-increase-parallelism=6
./build/clearctrl_bench --log-path=logs/controller.csv --controller=on --runner-schedule="1000:100:300;4000:100:300;500:100:300"  

source .venv/bin/activate
python analyze_logs.py logs/controller.csv logs/fixed2.csv logs/fixed4.csv logs/fixed6.csv
deactivate
