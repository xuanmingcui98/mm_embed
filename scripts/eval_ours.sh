#!/bin/bash
#SBATCH --job-name=evalchat
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH -t 1-0
#SBATCH --output=slurm_logs/eval/all_%j.out
#SBATCH --account=all

source /home/xuanmingcui/miniconda3/etc/profile.d/conda.sh
conda activate def

# ==============================================================================
# Configuration
# ==============================================================================
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
BATCH_SIZE=8
# MODALITIES=("image" "video" "visdoc")
MODALITIES=("image" "video" "visdoc")  

checkpoint_path="runs/qwen2-2b_v2_chat_lr2e-4_bs128_numworkerssharding_maxsteps625_8x8_473793"

# Loop through each modality for the current model
for MODALITY in "${MODALITIES[@]}"; do
  DATA_CONFIG_PATH="configs/eval/$MODALITY.yaml"

configs/eval
  echo "-------------------------------------------------"
  echo "  - Modality: $MODALITY"

  cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=8 --master_port=2277 --max_restarts=0 eval.py \
    --per_device_eval_batch_size $BATCH_SIZE \
    --checkpoint_path \"$checkpoint_path\" \
    --dataset_config \"$DATA_CONFIG_PATH\" 2>&1 | tee $checkpoint_path/eval_${MODALITY}.log"

  echo "  - Executing command..."
  # echo "$cmd" # Uncomment for debugging the exact command
  eval "$cmd"
  echo "  - Done."
  echo "-------------------------------------------------"
done

python scripts/eval_utils/get_result_summary.py --dir "$checkpoint_path/eval_ours"

echo "✅ All jobs completed."
