set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=""
PYTHON_MODE="posix"

if [ -r /proc/version ] && grep -qi microsoft /proc/version; then
  PYTHON_MODE="wsl"
fi

if [ -f "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [ -f "$ROOT/.venv/Scripts/python.exe" ]; then
  PYTHON="$ROOT/.venv/Scripts/python.exe"
  if [ "$PYTHON_MODE" = "wsl" ]; then
    PYTHON_MODE="windows"
  fi
else
  echo "Python not found in .venv. Expected .venv/Scripts/python.exe or .venv/bin/python." >&2
  exit 1
fi

to_python_path() {
  if [ "$PYTHON_MODE" = "windows" ]; then
    wslpath -w "$1"
  else
    printf "%s" "$1"
  fi
}

run_pipeline() {
  local script="$1"
  local feature_set="$2"
  "$PYTHON" "$(to_python_path "$ROOT/src/$script")" --feature-set "$feature_set"
}

pids=()
for feature_set in set1 set2 set4; do
  run_pipeline "ahc_pipeline.py" "$feature_set" &
  pids+=($!)
  run_pipeline "dbscan_pipeline.py" "$feature_set" &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [ "$status" -ne 0 ]; then
  echo "One or more pipeline runs failed; skipping report ranking." >&2
  exit "$status"
fi

"$PYTHON" "$(to_python_path "$ROOT/src/cluster_report_ranker.py")"
