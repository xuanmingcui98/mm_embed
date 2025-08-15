import json
import sys

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoProcessor, AutoConfig

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

def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else: 
            _batch[key] = value
    return _batch

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']


def parse_eval_args():
    parser = argparse.ArgumentParser(description="Evaluation arguments")
    parser.add_argument("--sft_model_checkpoint_path", type=str, default=None, help="Path to the SFT model checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--subset_name", type=str, nargs='+', required=True, help="List of subset names to evaluate")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--image_dir", type=str, default="/home/xuanmingcui/datasets/MMEB-eval/eval_images/")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to use for evaluation")
    parser.add_argument("--dataset_name", type=str, default="/home/xuanmingcui/datasets/MMEB-eval", help="Name of the dataset to evaluate")

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
        if not os.path.exists(os.path.join(eval_args.sft_model_checkpoint_path, js)) and "checkpoint-" in eval_args.sft_model_checkpoint_path:
            # we have to one level up to get the args
            arg_path = os.path.dirname(eval_args.sft_model_checkpoint_path)
        else:
            arg_path = eval_args.sft_model_checkpoint_path
        args.update(json.load(open(os.path.join(arg_path, js), 'r')))

    args = args | vars(eval_args)  
    model_args, data_args, training_args = parser.parse_dict(args)
        
    model_args.checkpoint_path = eval_args.sft_model_checkpoint_path
    output_path = os.path.join(model_args.sft_model_checkpoint_path, "eval_twostage_gen_emb")
    os.makedirs(output_path, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        num_crops=model_args.num_crops,
    )

    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=hf_config)
    setattr(training_args, 'model_backbone', model_backbone)
    setattr(model_args, 'model_backbone', model_backbone)

    print(f'model_args: {json.dumps(vars(model_args), indent=2)}')
    print(f'data_args: {json.dumps(vars(data_args), indent=2)}')

    model, processor = MMEBModel.load(model_args)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)


    qry_eval_collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
        apply_chat_template=data_args.apply_chat_template,
        mode="query"
    )

    tgt_eval_collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
        apply_chat_template=data_args.apply_chat_template_target,
        mode="target"
    )


    os.makedirs(os.path.join(eval_args.sft_model_checkpoint_path, "generation"), exist_ok=True)
    for idx, subset in enumerate(data_args.subset_name):

        os.makedirs(os.path.join(eval_args.sft_model_checkpoint_path, "generation", subset, "cot"), exist_ok=True)
        if not os.path.exists(os.path.join(eval_args.sft_model_checkpoint_path, "generation", subset, "cot", "query.pkl")):

            eval_qry_dataset = EvalDataset(
                data_args=data_args,
                model_args=model_args,
                subset=subset,
                text_field="qry_text",
                img_path_field="qry_img_path",
            )
            eval_qry_loader = DataLoader(
                eval_qry_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                collate_fn=qry_eval_collator,
                shuffle=False,
                drop_last=False,
                num_workers=training_args.dataloader_num_workers,
            )
        
            query_predictions = {}
            with torch.no_grad():
                for batch in tqdm(eval_qry_loader, desc="Encode query"):
                    batch = batch_to_device(batch, training_args.device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        answers, _ = model.generate(batch, return_hidden_states=False, return_decode_answer=True)
                        for i in range(len(answers)):
                            query_predictions[(batch['orig_text'][i], batch['img_path'][i])] = answers[i]
            
            pickle.dump(query_predictions, open(os.path.join(eval_args.sft_model_checkpoint_path, "generation", subset, "cot", "query.pkl"), "wb"))


        if not os.path.exists(os.path.join(eval_args.sft_model_checkpoint_path, "generation", subset, "cot", "target.pkl")):
            eval_tgt_dataset = EvalDataset(
                data_args=data_args,
                model_args=model_args,
                subset=subset,
                text_field="tgt_text",
                img_path_field="tgt_img_path",
            )

            eval_tgt_loader = DataLoader(
                eval_tgt_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                collate_fn=tgt_eval_collator,
                shuffle=False,

                num_workers=training_args.dataloader_num_workers,
            )
            target_predictions = {}
            with torch.no_grad():
                for batch in tqdm(eval_tgt_loader, desc="Encode target"):
                    batch = batch_to_device(batch, training_args.device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        answers, _ = model.generate(batch, return_hidden_states=False, return_decode_answer=True)
                        for i in range(len(answers)):
                            target_predictions[(batch['orig_text'][i], batch['img_path'][i])] = answers[i]
            
            pickle.dump(target_predictions, open(os.path.join(eval_args.sft_model_checkpoint_path, "generation", subset, "cot", "target.pkl"), "wb"))


    del model, processor


    ### part 2: Run embedding head with generated descriptions
    model_args.checkpoint_path = eval_args.checkpoint_path

    for js in ["model_args.json", "data_args.json", "training_args.json"]:
        if not os.path.exists(os.path.join(eval_args.checkpoint_path, js)) and "checkpoint-" in eval_args.checkpoint_path:
            # we have to one level up to get the args
            arg_path = os.path.dirname(eval_args.checkpoint_path)
        else:
            arg_path = eval_args.checkpoint_path
        args.update(json.load(open(os.path.join(arg_path, js), 'r')))

    args = args | vars(eval_args)  
    model_args, data_args, training_args = parser.parse_dict(args)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        num_crops=model_args.num_crops,
    )

    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=hf_config)
    setattr(training_args, 'model_backbone', model_backbone)
    setattr(model_args, 'model_backbone', model_backbone)

    print(f'model_args: {json.dumps(vars(model_args), indent=2)}')
    print(f'data_args: {json.dumps(vars(data_args), indent=2)}')

    model, processor = MMEBModel.load(model_args)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    # replace description with the generated query and target

    data_args.description_dir = os.path.join(model_args.sft_model_checkpoint_path, "generation")
    if data_args.target_description_dir:
        data_args.target_description_dir = os.path.join(model_args.sft_model_checkpoint_path, "generation")


    # ToDo: This part of code is a little bit hacky. Need to refactor later.
    for idx, subset in enumerate(data_args.subset_name):
        score_path = os.path.join(output_path, f"{subset}_score.json")

        if os.path.exists(score_path):
            try:
                with open(score_path, "r") as f:
                    score_dict = json.load(f)
                print(f"Found previous eval score, skipping {subset}")
                print(score_dict)
                continue
            except Exception as e:
                pass

        print(f"\033[91m{idx+1}/{len(data_args.subset_name)}: Processing {subset} now!\033[0m")
        encode_qry_path = os.path.join(output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(output_path, f"{subset}_tgt")


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

        print(f"{subset} qry length {len(eval_qry_dataset.paired_dataset)}")
        print(f"{subset} tgt length {len(eval_tgt_dataset.paired_dataset)}")


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

        if not os.path.exists(encode_qry_path) :
            encoded_tensor = []
            with torch.no_grad():
                print_rank(f"Encoding query for {subset} with {len(eval_qry_loader)} batches")
                for batch in eval_qry_loader:
                    batch = batch_to_device(batch, training_args.device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        output = model(qry=batch)
                    encoded_tensor.append(output["qry_reps"].cpu().detach().float().numpy())
            encoded_tensor = np.concatenate(encoded_tensor)
            with open(encode_qry_path, 'wb') as f:
                pickle.dump((encoded_tensor, eval_qry_dataset.paired_data), f)

        if not os.path.exists(encode_tgt_path):
            encoded_tensor = []
            print_rank(f"Encoding target for {subset} with {len(eval_tgt_loader)} batches")
            with torch.no_grad():
                for batch in eval_tgt_loader:
                    batch = batch_to_device(batch, training_args.device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        output = model(tgt=batch)
                    encoded_tensor.append(output["tgt_reps"].cpu().detach().float().numpy())
            encoded_tensor = np.concatenate(encoded_tensor)
            with open(encode_tgt_path, 'wb') as f:
                pickle.dump((encoded_tensor, eval_tgt_dataset.paired_data), f)

    for subset in data_args.subset_name:
        print(f"Calculating scores for {subset}...")
        score_path = os.path.join(output_path, f"{subset}_score.json")
        if os.path.exists(score_path):
            try:
                with open(score_path, "r") as f:
                    score_dict = json.load(f)
                print(f"Found previous eval score, skipping {subset}")
                print(score_dict)
                continue
            except Exception as e:
                pass
        encode_qry_path = os.path.join(output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(output_path, f"{subset}_tgt")
        with open(encode_qry_path, 'rb') as f:
            qry_tensor, qry_index = pickle.load(f)
        with open(encode_tgt_path, 'rb') as f:
            tgt_tensor, tgt_index = pickle.load(f)
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
            for tt in zip(row["tgt_text"], row["tgt_img_path"], [None] * len(row["tgt_text"])): # [row['qry_text'] if (subset in task_categories['vqa'].union(task_categories['classification'])) and data_args.apply_chat_template_target else None] * len(row["tgt_text"])):
                tgt_t.append(tgt_dict[tt])
                all_candidates.append(tt)
            tgt_t = np.stack(tgt_t, axis=0)  # (num_candidate, dim)
            scores, pred = get_pred(qry_t, tgt_t, normalization=model_args.normalize)
            if pred == 0:
                n_correct += 1
            all_pred.append(all_candidates[pred])
        with open(os.path.join(output_path, f"{subset}_pred.txt"), "w") as f:
            for item in all_pred:
                f.write(f"{item}\n")
        
        print(f"Outputting final score to: {score_path}")
        with open(score_path, "w") as f:
            score_dict = {"acc": n_correct/len(eval_data), "num_correct": n_correct, "num_pred": len(eval_data)}
            json.dump(score_dict, f, indent=4)
        print(f"\033[91m{subset} accuracy: {n_correct/len(eval_data)}\033[0m")

        # delete all the _qry and _tgt files
        os.remove(encode_qry_path)
        os.remove(encode_tgt_path)


if __name__ == "__main__":
    main()
