#! /bin/bash
set -euo pipefail

SCHEDULE="1000:100:300;4000:100:300;500:100:300"
# Default --ctrl-bandit-arms: "2:1;4:1;4:2;6:2;6:4"
#   index 0 = (2,1), index 1 = (4,1), index 3 = (6,2)
./build/clearctrl_bench --log-path=logs/fixed2.csv --controller=off --runner-schedule="$SCHEDULE" --ctrl-bandit-initial-arm=0 --rocksdb-increase-parallelism=6
./build/clearctrl_bench --log-path=logs/fixed4.csv --controller=off --runner-schedule="$SCHEDULE" --ctrl-bandit-initial-arm=1 --rocksdb-increase-parallelism=6
./build/clearctrl_bench --log-path=logs/fixed6.csv --controller=off --runner-schedule="$SCHEDULE" --ctrl-bandit-initial-arm=3 --rocksdb-increase-parallelism=6
./build/clearctrl_bench --log-path=logs/controller.csv --controller=on --runner-schedule="$SCHEDULE"

source .venv/bin/activate
python analyze_logs.py logs/controller.csv logs/fixed2.csv logs/fixed4.csv logs/fixed6.csv
deactivate
