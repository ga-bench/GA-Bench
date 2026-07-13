#!/bin/bash
# Launch GROBID Singularity container on the local node.

set -u

GROBID_SIF="$1"
GROBID_LOG="$2"

TMP_DIR="/tmp/grobid_tmp_$USER"
mkdir -p "$TMP_DIR"
chmod 1777 "$TMP_DIR"

pkill -f 'grobid-service' 2>/dev/null || true
sleep 2

: > "$GROBID_LOG"

setsid singularity run \
    --pwd /opt/grobid \
    --bind "$TMP_DIR":/opt/grobid/grobid-home/tmp \
    "$GROBID_SIF" \
    </dev/null >"$GROBID_LOG" 2>&1 &

GROBID_PID=$!
disown $GROBID_PID 2>/dev/null || true

sleep 2

if kill -0 $GROBID_PID 2>/dev/null; then
    echo "OK: GROBID started, pid=$GROBID_PID, log=$GROBID_LOG"
    exit 0
else
    echo "FAIL: GROBID died immediately. Last 50 lines of log:"
    tail -50 "$GROBID_LOG"
    exit 1
fi