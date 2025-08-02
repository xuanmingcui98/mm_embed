#!/bin/bash

#SBATCH --job-name=2b_chat
#SBATCH --nodes=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH -t 2-0
#SBATCH --account=all

set -e

NUM_PROCESSES=$(expr $SLURM_NNODES \* $SLURM_GPUS_ON_NODE)
export TOKENIZERS_PARALLELISM=true

echo "NNODES: $SLURM_NNODES"
echo "NUM_PROCESSES: $NUM_PROCESSES"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6002

export LAUNCHER="
    accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $SLURM_NNODES \
    --mixed_precision bf16 \
    "
source /home/xuanmingcui/miniconda3/etc/profile.d/conda.sh
conda activate def

EXP_NAME="qwen2-2b_v2_imageonly_chat_qry_tgt_sft_lr2e-4_4x8_${SLURM_JOB_ID}"
EXP_DIR="runs/$EXP_NAME"

mkdir -p $EXP_DIR

PROGRAM="\
     train_sft.py \
     --query_description_dir descriptions --target_description_dir descriptions_target \
     --do_sft_query True --do_sft_target True \
     --do_cl False \
     --apply_chat_template True \
     --lora --lora_r 16 --model_name Qwen/Qwen2-VL-2B-Instruct --bf16 --pooling_module eos --normalize True \
     --temperature 0.02 --dataloader_num_workers 8 --dataset_config configs/train/train_image.yaml \
     --run_name $EXP_NAME --output_dir $EXP_DIR --grad_cache False --per_device_train_batch_size 4 \
     --gc_q_chunk_size 8 --gc_p_chunk_size 8 --interleave_batch_size 0.0625 --lr_scheduler_type linear \
     --gradient_accumulation_steps 8 \
     --learning_rate 2e-4 --num_train_epochs 1 --warmup_ratio 0.05 --save_steps 100 --logging_steps 1 \
     --save_safetensors True --remove_unused_columns False --resume_from auto --report_to wandb 2>&1 | tee $EXP_DIR/train.log"

export CMD="$LAUNCHER $PROGRAM"

srun --jobid $SLURM_JOBID bash -c "$CMD"
