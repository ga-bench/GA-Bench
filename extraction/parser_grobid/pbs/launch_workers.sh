#!/bin/bash
# Launches N worker.py processes on the local node, detached.

set -x

source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/python-3.13.5 2>&1
module load java/jdk-17 2>&1

echo "=== launch_workers.sh START ==="
echo "PWD: $(pwd)"
echo "USER: $USER"
echo "HOSTNAME: $(hostname)"
echo "Arg count: $#"

if [ $# -lt 9 ]; then
    echo "FAIL: need at least 9 args, got $#"
    exit 2
fi

N_WORKERS="$1"
PYTHON="$2"
SRC_DIR="$3"
GROBID_URL="$4"
WORK_QUEUE="$5"
NODE_ID="$6"
JAR_PATH="$7"
LOG_DIR="$8"
WORKER_LOG_PREFIX="$9"
MAX_PAPERS="${10:-}"

echo "N_WORKERS=$N_WORKERS"
echo "PYTHON=$PYTHON"
echo "SRC_DIR=$SRC_DIR"
echo "GROBID_URL=$GROBID_URL"
echo "WORK_QUEUE=$WORK_QUEUE"
echo "NODE_ID=$NODE_ID"
echo "JAR_PATH=$JAR_PATH"
echo "LOG_DIR=$LOG_DIR"
echo "WORKER_LOG_PREFIX=$WORKER_LOG_PREFIX"
echo "MAX_PAPERS=${MAX_PAPERS:-<unset>}"

if [ ! -x "$PYTHON" ]; then
    echo "FAIL: PYTHON not executable: $PYTHON"
    ls -lh "$PYTHON" 2>&1
    exit 3
fi

if [ ! -d "$SRC_DIR" ]; then
    echo "FAIL: SRC_DIR not directory: $SRC_DIR"
    exit 4
fi

if [ ! -f "$SRC_DIR/worker.py" ]; then
    echo "FAIL: worker.py not found: $SRC_DIR/worker.py"
    exit 5
fi

if [ ! -d "$WORK_QUEUE" ]; then
    echo "FAIL: WORK_QUEUE directory not found: $WORK_QUEUE"
    exit 6
fi

if [ ! -f "$JAR_PATH" ]; then
    echo "FAIL: JAR_PATH not found: $JAR_PATH"
    exit 7
fi

mkdir -p "$LOG_DIR"

cd "$SRC_DIR" || exit 8

echo "=== Python check ==="
"$PYTHON" -c "import sys; print(sys.version)" || exit 10

echo "=== Launching workers ==="

for WID in $(seq 0 $((N_WORKERS - 1))); do
    WID_PADDED=$(printf "%02d" "$WID")
    LOG_FILE="${LOG_DIR}/${WORKER_LOG_PREFIX}_${NODE_ID}_w${WID_PADDED}.log"

    echo "Launching worker numeric_id=$WID log=$LOG_FILE"

    if [ -n "$MAX_PAPERS" ]; then
        setsid "$PYTHON" "$SRC_DIR/worker.py" loop \
            --grobid-url "$GROBID_URL" \
            --work-queue "$WORK_QUEUE" \
            --worker-id "$WID" \
            --node-id "$NODE_ID" \
            --jar-path "$JAR_PATH" \
            --max-papers "$MAX_PAPERS" \
            > "$LOG_FILE" 2>&1 < /dev/null &
    else
        setsid "$PYTHON" "$SRC_DIR/worker.py" loop \
            --grobid-url "$GROBID_URL" \
            --work-queue "$WORK_QUEUE" \
            --worker-id "$WID" \
            --node-id "$NODE_ID" \
            --jar-path "$JAR_PATH" \
            > "$LOG_FILE" 2>&1 < /dev/null &
    fi

    sleep 0.5
done

echo "=== launch_workers.sh DONE ==="
exit 0