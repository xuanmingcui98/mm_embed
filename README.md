Codebase built on top of [VLM2Vec V2](https://github.com/TIGER-AI-Lab/VLM2Vec).

### Installation:
```
$ git clone git@github.com:xuanmingcui98/mm_embed.git
```

To setup environment, for aws cluster, you may want to follow the [internal instruction](https://www.internalfb.com/wiki/Cloud/Cloud_HPC/High_Performance_Computing_as_a_Service/User_Guide_0/Set_up_a_Conda_environment/) to use the aws distribution for torch. For other packages, please refer to ```requirements.txt```

### Data 

#### Download

To download data, please refer to `experiments/public/data/download_data.sh`. The script automatically unzip image partition. For video, visdoc, and their evaluation data, we have to manually unzip.

#### Data configurations

First set up default config from original VLM2Vec:

```
# at root level
mkdir configs; cd configs; mkdir train; cd train; wget https://github.com/TIGER-AI-Lab/VLM2Vec/blob/main/experiments/public/train/train_alltasks.yaml; cd ..
mkdir eval; cd eval
wget https://github.com/TIGER-AI-Lab/VLM2Vec/blob/main/experiments/public/eval/image.yaml
wget https://github.com/TIGER-AI-Lab/VLM2Vec/blob/main/experiments/public/eval/video.yaml
wget https://github.com/TIGER-AI-Lab/VLM2Vec/blob/main/experiments/public/eval/visdoc.yaml
cd ../..
```

Then change the data paths in the configs accordingly.


### Run & Eval

To launch multinode training on slurm, please refer to `scripts/train_full.sh`, and change the path/name of the environment accordingly. To run baseline, please explicitly set `--apply_chat_template False`.

For evaluation, for local evaluation, please refer to eval.py. For multi-gpu eval, please refer to `scripts/eval_v1.sh`, and provide corresponding checkpoint_path.

Checkpoints: [google drive](https://drive.google.com/drive/folders/1qPzMIEtTQufo0F8mEdX6Z6ViyjgUVZIF?dmr=1&ec=wgc-drive-globalnav-goto)
ECRs: [google drive](https://drive.google.com/drive/folders/1TlPSwth76yu_tE_QqxMrarDBayx-R__h?dmr=1&ec=wgc-drive-globalnav-goto)





