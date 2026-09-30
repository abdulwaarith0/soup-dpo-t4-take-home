#!/usr/bin/env bash
# Usage: bash scripts/run_logged.sh NAME -- command [args...]
#
# Runs one phase with raw evidence kept:
#   logs/t4/NAME.log            every output line, UTC-timestamped (scripts/ts.py)
#   logs/t4/NAME.exit           the command's exit code
#   nvidia-smi/NAME.csv         GPU memory / utilisation / clock / power every 500 ms
#   nvidia-smi/NAME_full.txt    a full `nvidia-smi` screen every 60 s, with the time
set -u -o pipefail
name=$1; shift
[ "${1:-}" = "--" ] && shift
mkdir -p logs/t4 nvidia-smi

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,memory.total,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate \
  --format=csv -lms 500 > "nvidia-smi/$name.csv" 2>&1 &
csv_pid=$!
( while true; do echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ)"; nvidia-smi; sleep 60; done ) > "nvidia-smi/${name}_full.txt" 2>&1 &
full_pid=$!

echo "\$ $*" | python scripts/ts.py "logs/t4/$name.log" > /dev/null
PYTHONUNBUFFERED=1 "$@" 2>&1 | python scripts/ts.py "logs/t4/$name.log"
code=${PIPESTATUS[0]}

kill "$csv_pid" "$full_pid" 2>/dev/null
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) (end)" >> "nvidia-smi/${name}_full.txt"
nvidia-smi >> "nvidia-smi/${name}_full.txt" 2>&1
echo "$code" > "logs/t4/$name.exit"
echo "[$name] exit $code"
exit "$code"
