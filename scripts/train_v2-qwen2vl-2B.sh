#!/bin/bash


#SBATCH --job-name=2b
#SBATCH --nodes=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH -t 2-0
#SBATCH --output=slurm_logs/full/train_mmebv2_qwen2-2b_bl_1x8_%j.out
#SBATCH --account=all

# NOTE: replace ... with actual paths

# export HF_DATASETS_CACHE=...
# export HF_HOME=...
# export WANDB_DISABLED=false
# export WANDB_PROJECT=...
# export WANDB_API_KEY=...
# export HUGGING_FACE_HUB_TOKEN=...
# export WANDB_PROJECT=VLM2VecV2
# export WANDB_RUN_GROUP=...
export EXP_NAME=Qwen2vl_2B.image+visdoc+video.autoresize.lora16.BS1024.IB64.GCq8p8.NormTemp002.lr5e5.step5kwarm100.8H100

export WANDB_NAME=$EXP_NAME
export EXP_DIR=/home/xuanmingcui/projects/vlm2vec_orig/VLM2Vec/runs/$EXP_NAME
export WANDB_DIR=$EXP_DIR
echo $EXP_DIR

mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*

cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2207 --max_restarts=0 train.py --lora --lora_r 16 --model_name Qwen/Qwen2-VL-2B-Instruct --bf16 --pooling eos --normalize True --temperature 0.02 --dataloader_num_workers 8 --dataset_config experiments/release/train/train_alltasks.yaml --run_name $EXP_NAME --output_dir $EXP_DIR --grad_cache True --per_device_train_batch_size 128 --gc_q_chunk_size 8 --gc_p_chunk_size 8 --interleave_batch_size 64 --lr_scheduler_type linear --learning_rate 5e-5 --max_steps 5000 --warmup_steps 100 --save_steps 50 --logging_steps 1 --save_safetensors True --remove_unused_columns False --resume_from auto --report_to wandb 2>&1 | tee $EXP_DIR/train.log"

echo $cmd
srun bash -c $cmd
