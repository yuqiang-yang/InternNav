export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export NCCL_SOCKET_IFNAME=bond0 
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5
export TRITON_CACHE_DIR=/tmp/zhuchenming/.triton
export HF_HOME=/mnt/inspurfs/efm_t/zhuchenming

look_down_deg="_30deg"
proj_height="_goals_height_0"
PROMPT_VERSION="qwenvl_2_5"
pixel="_coord"

MID_RUN_NAME="vln_one_stage_with_qa_bs256_backup-checkpoint-21000"

srun -p efm_t \
    --gres=gpu:8 \
    --ntasks=8 \
    -x HOST-10-140-66-68,HOST-10-140-66-182,HOST-10-140-66-181 \
    --time=0-20:00:00 \
    --ntasks-per-node=8 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    python scripts/eval/eval_habitat.py \
    --model_path /mnt/inspurfs/efm_t/zhuchenming/release/ckpt/${MID_RUN_NAME} \
    --mode system2 \
    --output_path results/$MID_RUN_NAME/val_unseen \