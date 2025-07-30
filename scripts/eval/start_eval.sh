#!/bin/bash
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate internutopia

source /root/miniconda3/etc/profile.d/conda.sh
conda activate internutopia

CONFIG=scripts/eval/configs/h1_cma_cfg.py

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "unknown parameter: $1"
            exit 1
            ;;
    esac
done
processes=$(ps -ef | grep 'internnav/agent/utils/server.py' | grep -v grep | awk '{print $2}')
if [ -n "$processes" ]; then
    for pid in $processes; do
        kill -9 $pid
        echo "kill: $pid"
    done
fi
python internnav/agent/utils/server.py --config $CONFIG > server.log 2>&1 &


RETRY_LIMIT=5
MONITOR_INTERVAL=60
DEADLOCK_THRESHOLD=$((5 * 60))

START_COMMAND="python -u scripts/eval/eval.py --config $CONFIG"
LOG_FILE="eval.log"

pid=0

retry_count=0


start_process() {
    echo "Starting process..."
    $START_COMMAND > "$LOG_FILE" 2>&1 &
    pid=$!
}


check_process() {
    if ! kill -0 $pid > /dev/null 2>&1; then
        echo "Process $pid has exited."
        return 1
    fi
    return 0
}


check_log_update() {
    if [ ! -e "$LOG_FILE" ]; then
        return 1
    fi
    last_update=$(stat -c %Y "$LOG_FILE")
    current_time=$(date +%s)

    delta=$(( current_time - last_update ))

    if [ $delta -ge $DEADLOCK_THRESHOLD ]; then
        echo "Log file has not been updated for $((DEADLOCK_THRESHOLD / 60)) minutes."
        return 1
    fi

    return 0
}

start_process

while true; do
    sleep $MONITOR_INTERVAL
    echo "start healthcheck"

    if ! check_process; then
        if [ $retry_count -lt $RETRY_LIMIT ]; then
            echo "Retrying... (Attempt $((retry_count + 1))/$RETRY_LIMIT)"
            retry_count=$((retry_count + 1))
            start_process
        else
            echo "Exceeded maximum retry attempts. Exiting."
            exit 1
        fi
    else
        if ! check_log_update; then
            if [ $retry_count -lt $RETRY_LIMIT ]; then
                echo "Restarting process due to log file not updating... (Attempt $((retry_count + 1))/$RETRY_LIMIT)"
                retry_count=$((retry_count + 1))
                kill -9 $pid
                start_process
            else
                echo "Exceeded maximum retry attempts. Exiting."
                exit 1
            fi
        fi
    fi
done