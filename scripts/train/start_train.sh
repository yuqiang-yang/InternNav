#!/bin/bash

# 默认值
NAME=20250723_navdp_train_debug
MODEL=navdp

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --name)
            NAME="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 设置GPU设备和NUM_GPUS
case $MODEL in
    "rdp")
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        NUM_GPUS=4
        ;;
    "cma")
        export CUDA_VISIBLE_DEVICES=0
        NUM_GPUS=1
        ;;
    "seq2seq")
        export CUDA_VISIBLE_DEVICES=0
        NUM_GPUS=1
        ;;
    "navdp")
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        NUM_GPUS=8
        ;;
    *)
        echo "错误: 不支持的模型类型: $MODEL"
        exit 1
        ;;
esac

# 检查NUM_GPUS是否已设置
if [[ -z $NUM_GPUS ]]; then
    echo "错误: 未设置 NUM_GPUS"
    exit 1
fi


echo "使用 torchrun 启动 $MODEL 训练，使用 $NUM_GPUS 块 GPU (CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES)"
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CPP_LOG_LEVEL=INFO
export NCCL_DEBUG=INFO
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12345 \
    scripts/train/train.py \
    --name "$NAME" \
    --model-name "$MODEL"