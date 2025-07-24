export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export NCCL_SOCKET_IFNAME=bond0 
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5
export TRITON_CACHE_DIR=/tmp/zhuchenming/.triton
export HF_HOME=/mnt/inspurfs/efm_t/zhuchenming

look_down_deg="_30deg"
proj_height="_goals_height_0"
PROMPT_VERSION="qwenvl_2_5"
pixel="_coord"

MID_RUN_NAME="StreamVLN_Video_qwenvl_2_5_8history_look_down_30deg_goals_height_0_dit_select_size_1_predict_step_nums_32_scale_vln_go2_60cm_15deg-checkpoint-34812"

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
    --predict_step_nums 32 \
    --continuous_traj \
    --output_path results/$MID_RUN_NAME/val_unseen_32traj_8steps \