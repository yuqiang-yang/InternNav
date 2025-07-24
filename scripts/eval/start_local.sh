#!/bin/bash
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate grutopia

source /root/miniconda3/etc/profile.d/conda.sh
conda activate grutopia

CONFIG=scripts/eval/configs/h1_cma_cfg.py
GRUTOPIA_ASSETS_PATH=/shared/smartbot/datasets/GRUtopia-assets

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --grutopia_assets_path)
            GRUTOPIA_ASSETS_PATH="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done
processes=$(ps -ef | grep 'rnavigation/agent/utils/server.py' | grep -v grep | awk '{print $2}')
if [ -n "$processes" ]; then
    for pid in $processes; do
        kill -9 $pid
        echo "kill: $pid"
    done
fi
python grnavigation/agent/utils/server.py --config $CONFIG > server.log 2>&1 &


export GRUTOPIA_ASSETS_PATH=$GRUTOPIA_ASSETS_PATH
# 配置参数

RETRY_LIMIT=5    # 设置最大重试次数
MONITOR_INTERVAL=60 # 监控间隔（秒）
DEADLOCK_THRESHOLD=$((5 * 60)) # 判断卡死的时间 (秒)

# 启动进程的启动命令
START_COMMAND="python -u scripts/eval/eval.py --config $CONFIG" # 替换为实际的启动命令
LOG_FILE="eval.log"

# 进程 ID
pid=0
# 当前重试次数
retry_count=0

# 启动进程函数
start_process() {
    echo "Starting process..."
    $START_COMMAND > "$LOG_FILE" 2>&1 &
    pid=$!
}

# 检查进程状态函数
check_process() {
    if ! kill -0 $pid > /dev/null 2>&1; then
        echo "Process $pid has exited."
        return 1 # 表示进程已退出
    fi
    return 0 # 表示进程仍在运行
}

# 检查日志是否更新
check_log_update() {
    if [ ! -e "$LOG_FILE" ]; then
        return 1 # 文件未找到，视为未更新
    fi
    last_update=$(stat -c %Y "$LOG_FILE")
    current_time=$(date +%s)

    # 使用 $(( ... )) 进行算术运算
    delta=$(( current_time - last_update ))

    if [ $delta -ge $DEADLOCK_THRESHOLD ]; then
        echo "Log file has not been updated for $((DEADLOCK_THRESHOLD / 60)) minutes."
        return 1 # 日志未更新，视为卡死
    fi

    return 0 # 日志已更新
}

# 启动进程
start_process

while true; do
    sleep $MONITOR_INTERVAL
    echo "start healthcheck"

    # 检查进程状态
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
        # 进程仍在运行，检查日志
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