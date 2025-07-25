#!/bin/bash

# Default values
NAME=navdp_train_debug
MODEL=navdp

# Parse command line arguments
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
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Set GPU devices and NUM_GPUS
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
        echo "Error: Unsupported model type: $MODEL"
        exit 1
        ;;
esac

# Check if NUM_GPUS is set
if [[ -z $NUM_GPUS ]]; then
    echo "Error: NUM_GPUS is not set"
    exit 1
fi


echo "Using torchrun to start $MODEL training, using $NUM_GPUS GPUs (CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES)"
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