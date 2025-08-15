import json
import sys

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoProcessor, AutoConfig, AutoModel

from src.model import MMEBModel
from src.dataset import EvalDataset
from src.collator import EvalCollator
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
from datasets import load_dataset
from evaluation.eval_utils import get_pred
from src.utils import print_rank
from src.model_utils import get_backbone_name
import argparse
from src.prompts import task_categories

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch

def parse_eval_args():
    parser = argparse.ArgumentParser(description="Evaluation arguments")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--subset_name", type=str, nargs='+', required=True, help="List of subset names to evaluate")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--image_dir", type=str, default="/home/xuanmingcui/datasets/MMEB-eval/eval_images/")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to use for evaluation")
    parser.add_argument("--dataset_name", type=str, default="TIGER-Lab/MMEB-eval", help="Name of the dataset to evaluate")

    return parser.parse_args()

def main():
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)

    eval_args = parse_eval_args()

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    args = {}
    for js in ["model_args.json", "data_args.json", "training_args.json"]:
        args.update(json.load(open(os.path.join(eval_args.checkpoint_path, js), 'r')))

    args = args | vars(eval_args)  
    model_args, data_args, training_args = parser.parse_dict(args)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        num_crops=model_args.num_crops,
    )

    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=hf_config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')
    print(f'model_args: {json.dumps(vars(model_args), indent=2)}')
    print(f'data_args: {json.dumps(vars(data_args), indent=2)}')

    model = MMEBModel.load(model_args, processor)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    output_path = os.path.join(model_args.checkpoint_path, "eval_sft")
    os.makedirs(output_path, exist_ok=True)

    # jina v3 embedding collator

    class TargetCollator:

        def __call__(self, examples):
            if "Answer: " not in examples[0][0]:
                return [x[0] for x in examples]
            else:
                return [x[0].split("\n")[2].replace("Answer: ", "").strip(".") for x in examples]

    qry_eval_collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
        apply_chat_template=data_args.apply_chat_template,
    )

    tgt_eval_collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
        apply_chat_template=data_args.apply_chat_template_target,
    )

    # ToDo: This part of code is a little bit hacky. Need to refactor later.
    for idx, subset in enumerate(data_args.subset_name):

        print(f"\033[91m{idx+1}/{len(data_args.subset_name)}: Processing {subset} now!\033[0m")
        # if os.path.exists(encode_qry_path) and os.path.exists(encode_tgt_path):
        #     continue
        setattr(model_args, 'model_backbone', 'qwen2_vl')
        eval_qry_dataset = EvalDataset(
            data_args=data_args,
            model_args=model_args,
            subset=subset,
            text_field="qry_text",
            img_path_field="qry_img_path",
        )
        eval_tgt_dataset = EvalDataset(
            data_args=data_args,
            model_args=model_args,
            subset=subset,
            text_field="tgt_text",
            img_path_field="tgt_img_path",
        )

        eval_qry_loader = DataLoader(
            eval_qry_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=qry_eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )
        eval_tgt_loader = DataLoader(
            eval_tgt_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=tgt_eval_collator,
            shuffle=False,
            num_workers=training_args.dataloader_num_workers,
        )

        annos = pickle.load(open("descriptions/MSCOCO_i2t/query/all.pkl", 'rb'))
        predicted_answers = []
        with torch.no_grad():
            for batch in tqdm(eval_qry_loader, desc="generate queries"):
                batch = batch_to_device(batch, training_args.device)
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                    # answers, _ = model.generate(batch, return_hidden_states=False, return_decode_answer=True)
                    # answers = [x.split("Answer: ")[-1].strip() for x in answers]
                    answers = [annos[(x['qry_text'], x['qry_img_path'])] for x in batch['paired_data']]
                predicted_answers.extend(answers)

        # predicted_answers = json.load(open("mscoco_i2t_pred.json", 'r'))

        # del model
        # torch.cuda.empty_cache()

        embedding_model_name = "google/siglip2-large-patch16-384"
        # embedding_model_name = "openai/clip-vit-base-patch32"
        embedding_model = AutoModel.from_pretrained(embedding_model_name, trust_remote_code=True)
        embedding_model = embedding_model.to(training_args.device)
        embedding_model.eval()
        processor = AutoProcessor.from_pretrained(embedding_model_name, trust_remote_code=True)

        qry_tensor = []

        with torch.no_grad():
            for i in tqdm(range(0, len(predicted_answers), 128), desc="Encode query"):
                # with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                inputs = processor(text=predicted_answers[i:i+128], padding=True, return_tensors="pt").to(training_args.device)
                output = embedding_model.get_text_features(**inputs)
                    # normalize
                    # output = output / output.norm(dim=-1, keepdim=True)
                qry_tensor.append(output.cpu().numpy())
        qry_tensor = np.concatenate(qry_tensor)

        tgt_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_tgt_loader, desc="Encode target"):
                # batch = batch_to_device(batch, training_args.device)
                # output = embedding_model.encode(batch, task="text-matching")

                if batch['texts'][0]:
                    text_inputs = processor(text=batch['texts'], padding=True, return_tensors="pt").to(training_args.device)
                    output = embedding_model.get_text_features(**text_inputs)


                if batch['images'][0] is not None:
                    image_inputs = processor(images=batch['images'], return_tensors="pt").to(training_args.device)
                    image_features = embedding_model.get_image_features(**image_inputs)
                    output = (output + image_features) / 2.0  # average text and image features
                
                # normalize
                output = output / output.norm(dim=-1, keepdim=True)

                tgt_tensor.append(output.cpu().numpy())
        tgt_tensor = np.concatenate(tgt_tensor)

        qry_index = eval_qry_dataset.paired_data
        tgt_index = eval_tgt_dataset.paired_data

    # for subset in tqdm(data_args.subset_name, desc="calculate score"):

        # def get_pred(q, t, normalization=True):
        #     """
        #     Calculate the similarity score between query and target tensors.
        #     Returns the index of the highest score.
        #     """
        #     if normalization:
        #         q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        #         t = t / np.linalg.norm(t, axis=-1, keepdims=True)
        #     # matmul
        #     scores = np.matmul(q, t.T)  # (1, num_candidate)
        #     pred = np.argmax(scores, axis=-1)  # (1,)
        #     return scores[0], pred
        
        qry_dict, tgt_dict = {}, {}
        for qry_t, tt in zip(qry_tensor, qry_index):
            text, img_path = tt["text"], tt["img_path"]
            qry_dict[(text, img_path)] = qry_t
        for tgt_t, tt in zip(tgt_tensor, tgt_index):
            text, img_path, query = tt["text"], tt["img_path"], tt["query"]
            tgt_dict[(text, img_path, query)] = tgt_t

        eval_dataset = EvalDataset(
            data_args=data_args,
            model_args=model_args,
            subset=subset,
            text_field="qry_text",
            img_path_field="qry_img_path",
        )
        eval_data = eval_dataset.eval_data

        n_correct = 0
        all_pred = []
        for row in eval_data:
            qry_t = qry_dict[(row["qry_text"], row["qry_img_path"])]  # (dim,)
            tgt_t, all_candidates = [], []
            for tt in zip(row["tgt_text"], row["tgt_img_path"], [row['qry_text'] if (subset in task_categories['vqa'].union(task_categories['classification'])) and data_args.apply_chat_template_target else None] * len(row["tgt_text"])):
                tgt_t.append(tgt_dict[tt])
                all_candidates.append(tt)
            tgt_t = np.stack(tgt_t, axis=0)  # (num_candidate, dim)
            scores, pred = get_pred(qry_t, tgt_t, normalization=model_args.normalize)
            if pred == 0:
                n_correct += 1
            all_pred.append(all_candidates[pred])
        print(f"\033[91m{subset} accuracy: {n_correct/len(eval_data)}\033[0m")
        with open(os.path.join(output_path, f"{subset}_pred.txt"), "w") as f:
            for item in all_pred:
                f.write(f"{item}\n")

        score_path = os.path.join(output_path, f"{subset}_sftonly_score.json")
        print(f"Outputting final score to: {score_path}")
        with open(score_path, "w") as f:
            score_dict = {"acc": n_correct/len(eval_data), "num_correct": n_correct, "num_pred": len(eval_data)}
            json.dump(score_dict, f, indent=4)


if __name__ == "__main__":
    main()
