#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH -t 1-0

#SBATCH --output=slurm_logs/eval/eval_qwen2-2b_v2_imageonly_chat_lr2e-4_8x8_471465_%j.out
#SBATCH --account=all

source /home/jianpengcheng/miniconda3/etc/profile.d/conda.sh
conda activate def

# ==============================================================================
# Configuration
# ==============================================================================
CUDA_VISIBLE_DEVICES="0,1,2,3"
BATCH_SIZE=16
# MODALITIES=("image" "video" "visdoc")
MODALITIES=("image")

checkpoint_path="/home/jianpengcheng/VLM2Vec/runs/qwen2-2b_v2_imageonly_chat_lr2e-4_ntokens16_8x8_471593"


# Loop through each modality for the current model
for MODALITY in "${MODALITIES[@]}"; do
  DATA_CONFIG_PATH="configs/eval/$MODALITY.yaml"

configs/eval
  echo "-------------------------------------------------"
  echo "  - Modality: $MODALITY"

  cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=4 --master_port=2277 --max_restarts=0 eval.py \
    --per_device_eval_batch_size $BATCH_SIZE \
    --checkpoint_path \"$checkpoint_path\" \
    --dataset_config \"$DATA_CONFIG_PATH\""

  echo "  - Executing command..."
  # echo "$cmd" # Uncomment for debugging the exact command
  eval "$cmd"
  echo "  - Done."
  echo "-------------------------------------------------"
done


echo "✅ All jobs completed."
