#!/bin/bash
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
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

if [ "$RANK" -eq 0 ]; then
    RAY_max_direct_call_object_size=104857600 ray start --head --port=6379
    sleep 20s
    ./scripts/eval/start_eval.sh --config $CONFIG
    sleep inf
else
    RAY_max_direct_call_object_size=104857600 ray start --address=${MASTER_ADDR}:6379
    sleep inf
fi
