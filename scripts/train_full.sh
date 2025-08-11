#!/bin/bash

#SBATCH --job-name=chat
#SBATCH --nodes=8
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH -t 1-0
#SBATCH --mem=0
#SBATCH --account=all

set -e

echo "NODELIST="${SLURM_NODELIST}
echo "SLURM_GPUS_ON_NODE="$SLURM_GPUS_ON_NODE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export MASTER_PORT=12341
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600

NNODES=$SLURM_NNODES
GPU_PER_NODE=$SLURM_GPUS_ON_NODE

source /home/xuanmingcui/miniconda3/etc/profile.d/conda.sh
conda activate def

EXP_NAME="qwen2-2b_v2_chat_lr2e-4_bs128_maxsteps625_8x8_${SLURM_JOB_ID}"
EXP_DIR="runs/$EXP_NAME"

rdzv_id=$RANDOM

mkdir -p $EXP_DIR

     # --use_symmetric_loss True \
     # --inter_task_temperature 0.2 \
    #  --meta_queries 16 \
srun torchrun --max_restarts=0 --nnodes $NNODES --nproc_per_node $GPU_PER_NODE --rdzv_id=$rdzv_id --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
     train.py \
     --apply_chat_template True \
     --lora --lora_r 16 --model_name Qwen/Qwen2-VL-2B-Instruct --bf16 --pooling_module eos --normalize True \
     --temperature 0.02 --dataloader_num_workers 2 --dataset_config configs/train/train_alltasks.yaml \
     --run_name $EXP_NAME --output_dir $EXP_DIR --grad_cache True --per_device_train_batch_size 128 \
     --gc_q_chunk_size 4 --gc_p_chunk_size 4 --interleave_batch_size 0.0625 --lr_scheduler_type linear \
     --learning_rate 2e-4 --max_steps 625 --warmup_ratio 0.05 --save_steps 100 --logging_steps 1 \
     --save_safetensors True --remove_unused_columns False --resume_from auto --report_to wandb 2>&1 | tee $EXP_DIR/train.log


if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "Training completed on master node. Submitting eval job..."
    sbatch --job-name=eval --output=slurm_logs/eval/eval_${run_name}.out --export=checkpoint_path=$EXP_DIR scripts/subscripts/eval_full.sh \

fi
