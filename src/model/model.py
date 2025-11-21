from typing import Dict, List
import torch
import os
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
from src.model.processor import QWEN2_5_VL_TOKENSELECTION
from src.arguments import ModelArguments, TrainingArguments

from src.arguments import ModelArguments
from src.model.processor import (LLAVA_NEXT, QWEN2_VL, PHI3V, 
                                 get_backbone_name, print_master, 
                                 QWEN2_5_VL, INTERNVIDEO2,
                                 QWEN2_VL_TOKENSELECTION, 
                                 backbone2model, GME, VLM_IMAGE_TOKENS, 
                                 QWEN3_VL,
                                 LamRA, LamRA_QWEN2_5, COLPALI, INTERNVL3, E5_V, PLM)
from src.model.baseline_backbone.colpali import ColPali
from src.model.baseline_backbone.gme.gme_inference import GmeQwen2VL
from src.model.baseline_backbone.lamra.lamra_inference import LamRAQwen2VL
from src.model.baseline_backbone.lamra.lamra_qwen25_inference import LamRAQwen25VL
from src.model.baseline_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
from src.model.baseline_backbone.llava_next import LlavaNextForConditionalGeneration
from src.model.processor import load_processor, get_visual_token_ids
from contextlib import nullcontext
from src.loss import SimpleContrastiveLoss, DistributedContrastiveLoss

from .pooler import (AttentionPooler, AttentionPoolingConfig, 
                     NVEmbedPooler, NVEmbedPoolingConfig,
                     MultilayerPatchedPooler, MultilayerPatchedPoolerConfig,
                     TruncatedSelfPooler, TruncatedSelfPoolerConfig,
                     MLPPoolingConfig, MLPPooler)

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']


class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 model_config,
                 encoder: PreTrainedModel,
                 processor,
                 pooling_module = None,
                 ):
        super().__init__()
        self.encoder_config = encoder.config
        self.model_config = model_config
        self.encoder = encoder
        self.pooling_module_type = model_config.pooling_module
        self.normalize = model_config.normalize
        self.temperature = torch.tensor(model_config.temperature)
        self.inter_task_temperature = torch.tensor(model_config.inter_task_temperature) if model_config.inter_task_temperature else None
        self.processor = processor
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()

        if self.model_config.learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(self.temperature))
            if model_config.inter_task_temperature is not None:
                self.inter_task_temperature = nn.Parameter(torch.tensor(model_config.inter_task_temperature))
            else:
                self.inter_task_temperature = None

        self.loss_fn = DistributedContrastiveLoss() if self.is_ddp else SimpleContrastiveLoss()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        if model_config.freeze_base_model:
            assert not model_config.lora, "lora is not supported when freeze_base_model is True"
            # freeze base model
            for param in encoder.parameters():
                param.requires_grad = False
            encoder.eval()

        self.pooling_module = pooling_module
        self.visual_token_ids = get_visual_token_ids(processor)

        if model_config.meta_queries is not None and model_config.meta_queries > 0:
            self.meta_queries = [f'<meta_query_{i}>' for i in range(model_config.meta_queries)]


    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.use_gradient_checkpointing = True
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            # unwrap if wrapped by DDP
            if hasattr(self.encoder, "module"):
                self.encoder.module.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            else:
                self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    @torch.no_grad()
    def generate(self, input, return_hidden_states=True, return_decode_answer=False):
        results = self.encoder.generate(**{
            'input_ids': input['input_ids'],
            'attention_mask': input['attention_mask'],
            'pixel_values': input.get('pixel_values', None),
            'image_grid_thw': input.get('image_grid_thw', None),
        }, max_new_tokens=512, return_dict_in_generate=True, output_hidden_states=True)

        generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(input['input_ids'], results['sequences'])
            ]

        decoded = hidden_states = None
        if return_decode_answer:
            decoded = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
            decoded = [d.split("Answer: ")[-1].strip() for d in decoded]

        if return_hidden_states:

            generated_ids_trimmed_masks = [i != self.processor.tokenizer.pad_token_id for i in generated_ids_trimmed]

            last_non_padded_index = [
                mask.int().cumsum(0).argmax().item() for mask in generated_ids_trimmed_masks
            ]

            hidden_states = [results.hidden_states[i][-1][j] if i != 0 else results.hidden_states[i][-1][j][-1].unsqueeze(0) for j, i in enumerate(last_non_padded_index)]
            hidden_states = torch.cat(hidden_states, dim=0)
        
        return decoded, hidden_states

    def encode_input(self, input, do_sft=False):
        if getattr(self, "model_backbone", None) == INTERNVIDEO2:
            if "input_ids" in input.keys():
                # text side
                text_output = self.encoder.get_text_encoder()(
                    input["input_ids"],
                    attention_mask=input["attention_mask"],
                    return_dict=True,
                    mode="text",
                )
                text_embeds = text_output.last_hidden_state
                pooled_text_embeds = text_embeds[:, 0]
                pooled_output = self.encoder.text_proj(pooled_text_embeds)
                pooled_output /= pooled_output.norm(dim=-1, keepdim=True)
                return pooled_output
            else:
                _, vfeat = self.encoder.encode_vision(input["pixel_values"], test=True)
                vfeat = self.encoder.vision_proj(vfeat)
                vfeat /= vfeat.norm(dim=-1, keepdim=True)
                return vfeat
        elif getattr(self, "model_backbone", None) in [GME, LamRA, LamRA_QWEN2_5]:
            # pooled_output = self.encoder(**input, return_dict=True, output_hidden_states=True)
            texts = [text.replace(VLM_IMAGE_TOKENS[QWEN2_VL] + '\n', '') for text in input["texts"]] # we are actually passing video queries so this should not happen
            images = []
            for imgs in input['images']:
                # if multi images are given, select the middle frame only
                if isinstance(imgs, list):
                    imgs = imgs[len(imgs) // 2]
                    assert not isinstance(imgs, list) # make sure we have extracted the middle frame and it is no longer a list
                    images.append(imgs)
                else:
                    images.append(imgs)
            pooled_output = self.encoder.get_fused_embeddings(texts=texts, images=images)
            return pooled_output
        elif getattr(self, "model_backbone", None) == COLPALI:
            pooled_output = self.encoder(**input, return_dict=True, output_hidden_states=True)
            return pooled_output
        elif getattr(self, "model_backbone", None) == LLAVA_NEXT:
            input['pixel_values'] = input['pixel_values'].squeeze(dim=1)
            input['image_sizes'] = input['image_sizes'].squeeze(dim=1)
            hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
            hidden_states = hidden_states.hidden_states[-1]
            pooled_output = self._pooling(hidden_states, input['attention_mask'])
            return pooled_output
        else:
            if not self.training and do_sft:
                _, hidden_states = self.generate(input, return_hidden_states=True, return_decode_answer=False)
            else:
                if self.model_config.freeze_base_model:
                    context = torch.no_grad()
                else:
                    context = nullcontext()
                with context:
                    results = self.encoder(**input, return_dict=True, output_hidden_states=True)
            hidden_states = results.hidden_states[-1]
            pooled_output = self._pooling(hidden_states, input['attention_mask'], input=input, results=results, do_sft=do_sft)
            return pooled_output, results

    def _pooling(self, last_hidden_state, attention_mask, input, results, do_sft=False):
        if self.pooling_module_type in {'last', 'eos', 'mlp'}:
            # if False: # debug
            if do_sft and not self.training:
                reps = last_hidden_state
            else:
                left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
                batch_size = last_hidden_state.shape[0]
                if left_padding:
                    if do_sft:
                        # when we do sft we need the second to last position 
                        # because llm's output is shifted by one
                        reps = last_hidden_state[torch.arange(batch_size), -2, :]
                    else:
                    # when we do sft we need the second to last position 
                    # because llm's output is shifted by one
                        reps = last_hidden_state[torch.arange(batch_size), -1, :]
                else:
                    # Calculate last 1 position in the original tensor
                    eos_indices = attention_mask.sum(dim=1) - 1
                    # Get the vectors at the last 1 position of each attention mask
                    reps = last_hidden_state[
                        torch.arange(batch_size, device=last_hidden_state.device), eos_indices]
            
            if self.pooling_module_type == 'mlp':
                # Apply MLP pooling
                reps = self.pooling_module(reps)

        elif self.pooling_module_type == 'attention_pooler':
            if self.model_config.mask_visual_tokens_for_pooling:
                # mask visual tokens
                for visual_token_id in self.visual_token_ids:
                    attention_mask[input['input_ids'] == visual_token_id] = 0

            reps = self.pooling_module(last_hidden_state, attention_mask)
        elif self.pooling_module_type in {"patched_pooler", "truncated_self_pooler"}:
            if self.model_config.mask_visual_tokens_for_pooling:
                # mask visual tokens
                for visual_token_id in self.visual_token_ids:
                    attention_mask[input['input_ids'] == visual_token_id] = 0
            reps = self.pooling_module(results.hidden_states, attention_mask)
        elif self.pooling_module_type == 'meta_queries':
            # concat meta queries

            assert hasattr(self, 'meta_queries'), "meta_queries is not set"
            meta_query_ids = self.processor.tokenizer.convert_tokens_to_ids(self.meta_queries)
            meta_query_id_idx = []
            for seq in input['input_ids']:
                indices = [(seq == meta_query_id).nonzero()[0][0].item() for meta_query_id in meta_query_ids]
                meta_query_id_idx.append(indices)
            
            meta_query_id_idx = torch.tensor(meta_query_id_idx, device=last_hidden_state.device)

            meta_query_hidden_states = torch.gather(
                last_hidden_state, dim=1, index=meta_query_id_idx.unsqueeze(-1).expand(-1, -1, last_hidden_state.shape[-1]))
            if self.model_config.meta_queries_aggregate_type == 'mean':
                # mean pooling
                reps = meta_query_hidden_states.mean(dim=1)
            elif self.model_config.meta_queries_aggregate_type == 'concat':
                # concat pooling
                reps = meta_query_hidden_states.cat(dim=1)
            elif self.model_config.meta_queries_aggregate_type == 'late_interaction':
                # late fusion pooling. no pooling 
                reps = meta_query_hidden_states
            elif self.model_config.meta_queries_aggregate_type == 'attention_pooler':
                # attention pooling for meta queries with n pooling_n_queries concatted along the latent dim
                reps = self.pooling_module(meta_query_hidden_states)
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    @classmethod
    def build(cls, model_args: ModelArguments, data_args, **kwargs):
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        setattr(model_args, 'model_backbone', model_backbone)
        print_master(f'Loading backbone [{model_backbone}] from {model_args.model_name}')
        
        processor = load_processor(model_args, data_args=data_args)

        # Loading the base model
        if model_backbone == PHI3V:
            config._attn_implementation = "eager"
            config.padding_side = "right"
            config.use_cache = False
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone == LLAVA_NEXT:
            config.use_cache = False
            config.padding_side = "left"
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone in [QWEN2_VL]:
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False
            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_args.model_backbone == QWEN2_5_VL:
            from transformers import Qwen2_5_VLForConditionalGeneration
            config._attn_implementation = "flash_attention_2" 
            # config._attn_implementation = "sdpa" 
            config.padding_side = "left"
            config.use_cache = False
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_name, 
                                                                            config=config,
                                                                            torch_dtype=torch.bfloat16)
        elif model_args.model_backbone == QWEN3_VL:
            from transformers import Qwen3VLForConditionalGeneration
            config._attn_implementation = "flash_attention_2" 
            # config._attn_implementation = "sdpa" 
            config.padding_side = "left"
            config.use_cache = False
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(model_args.model_name, 
                                                                            config=config,
                                                                            torch_dtype=torch.bfloat16)
        elif model_backbone in [PLM]:
            # config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False
            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )

        elif model_backbone in [QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION]:
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False

            from .utils import parse_layer_type
            lm_qwen_layer = 28
            vis_qwen_layer = 32
            lm_skip_layer = parse_layer_type(model_args.lm_skip_layer, lm_qwen_layer)
            vis_skip_layer = parse_layer_type(model_args.vis_skip_layer, vis_qwen_layer)

            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                lm_skip_layer=lm_skip_layer,
                vis_skip_layer=vis_skip_layer,
            )
        elif model_backbone == INTERNVL3:
            # config.use_cache = False
            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                attn_implementation="flash_attention_2",
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        else:
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name, **kwargs, config=config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)

        if model_args.base_lora_checkpoint_path:
            checkpoint_path = model_args.base_lora_checkpoint_path
            print(f'Loading existing lora adapter from {checkpoint_path} into base model')
            lora_config = LoraConfig.from_pretrained(checkpoint_path)
            # state_dict = torch.load(os.path.join(checkpoint_path, 'adapter_model.bin'), map_location='cpu')
            # if 'encoder.' in list(state_dict.keys())[0]:
            #     state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
            #     torch.save(state_dict, os.path.join(checkpoint_path, 'adapter_model.bin'))
            
            base_model = PeftModel.from_pretrained(base_model, checkpoint_path, config=lora_config, is_trainable=True)

            # base_model = lora_model.merge_and_unload()

        modules_to_save = None
        if model_args.meta_queries is not None and model_args.meta_queries > 0:

            processor.tokenizer.add_special_tokens(
                {'additional_special_tokens': [f'<meta_query_{i}>' for i in range(model_args.meta_queries)]})

            base_model.resize_token_embeddings(len(processor.tokenizer))
            modules_to_save = ["embed_tokens"]

        if model_args.lora and not model_args.base_lora_checkpoint_path:
            print_master(f'Loading lora adapter from {base_model}')
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(','),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False,
                modules_to_save=modules_to_save
            )
            base_model.enable_input_require_grads()
            base_model = get_peft_model(base_model, lora_config)

        if model_args.meta_queries is not None and model_args.meta_queries > 0:
            meta_query_ids = processor.tokenizer.convert_tokens_to_ids(
                [f'<meta_query_{i}>' for i in range(model_args.meta_queries)])
            embedding = base_model.get_input_embeddings()
            def grad_filter(grad):
                # @xuanming dirty way to not update the old embeddings. Since weight decay is 0 by default, this should be fine
                mask = torch.zeros_like(grad)
                mask[meta_query_ids, :] = 1.0
                return grad * mask

            embedding.weight.register_hook(grad_filter)


        pooling_module = None
        if model_args.pooling_module not in {'last', 'eos'}:
            print_master(f'Building attention pooling module')
            if model_args.pooling_module == 'attention_pooler':
                pooling_module = AttentionPooler(AttentionPoolingConfig(input_embed_dim=base_model.config.hidden_size, 
                                                                        output_embed_dim=base_model.config.hidden_size,
                                                                        n_queries=model_args.pooling_n_queries,
                                                                        n_head=model_args.pooling_n_heads,))
            elif model_args.pooling_module == 'nv_embed_pooler':
                pooling_module = AttentionPooler(AttentionPoolingConfig(hidden_dim=base_model.config.hidden_size,
                                                                                            latent_dim=base_model.config.hidden_size))
            elif model_args.pooling_module == "mlp":
                pooling_module = MLPPooler(MLPPoolingConfig(input_embed_dim=base_model.config.hidden_size, 
                                                            output_embed_dim=base_model.config.hidden_size))
            elif model_args.pooling_module == "patched_pooler":
                pooling_module = MultilayerPatchedPooler(
                    MultilayerPatchedPoolerConfig(
                    input_embed_dim=base_model.config.hidden_size,
                    output_embed_dim=base_model.config.hidden_size,
                    n_queries=model_args.pooling_n_queries,
                    num_layers=model_args.num_pooling_layers,
                    last_n_layers=model_args.pooling_last_n_layers)
                )
            elif model_args.pooling_module == "truncated_self_pooler":
                pooling_module = TruncatedSelfPooler(
                    base_model,
                    TruncatedSelfPoolerConfig(
                        num_layers=model_args.num_pooling_layers,
                        last_n_layers=model_args.pooling_last_n_layers)
                )
            elif model_args.meta_queries is not None \
                and model_args.meta_queries > 0 \
                and model_args.meta_queries_aggregate_type == "attention_pooler":
                # attention pooling for meta queries with n pooling_n_queries concatted along the latent dim
                pooling_module = AttentionPooler(AttentionPoolingConfig(input_embed_dim=base_model.config.hidden_size,
                                                                        output_embed_dim=base_model.config.hidden_size,
                                                                        n_queries=model_args.pooling_n_queries,
                                                                        n_head=model_args.pooling_n_heads,
                                                                        aggregate="concat"))


        model = cls(
            encoder=base_model,
            model_config=model_args,
            pooling_module=pooling_module,
            processor=processor)

        return model, processor


    @classmethod
    def load(cls, model_args: ModelArguments, data_args, is_trainable=True, **kwargs):
        # Loading the base model
        model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
            model_backbone = get_backbone_name(hf_config=config, model_type=model_args.model_type)
            setattr(model_args, 'model_backbone', model_backbone)

        processor = load_processor(model_args, data_args=data_args)
        print_master(f'Loading backbone [{model_args.model_backbone}] from {model_name_or_path}')
        if model_args.model_backbone in {LLAVA_NEXT, QWEN2_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, E5_V}:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config._attn_implementation = "flash_attention_2"
            config.vision_config._attn_implementation = "flash_attention_2"
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                config=config
            )
        elif model_args.model_backbone == QWEN2_5_VL:
            from transformers import Qwen2_5_VLForConditionalGeneration
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_name, 
                                                                            trust_remote_code=True,
                                                                            torch_dtype=torch.bfloat16)

        elif model_args.model_backbone == PHI3V:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            config.padding_side = "right"
            base_model = Phi3VForCausalLM.from_pretrained(model_args.model_name, **kwargs, config=config,
                                                          torch_dtype=torch.bfloat16, trust_remote_code=True)
            base_model.padding_side = "right"
        elif model_args.model_backbone == INTERNVIDEO2:
            print_master(f'Loading backbone [{model_args.model_backbone}] from {"src/model/vlm_backbone/internvideo2/"}')
            config = AutoConfig.from_pretrained("src/model/vlm_backbone/internvideo2/",
                                                trust_remote_code=True)
            base_model = backbone2model[model_args.model_backbone].from_pretrained("src/model/vlm_backbone/internvideo2/", config=config,
                                                                                   trust_remote_code=True)
        elif model_args.model_backbone == GME:
            base_model = GmeQwen2VL(model_args.model_name, processor=kwargs['processor'])
            setattr(base_model, 'config', config)
        elif model_args.model_backbone == LamRA:
            base_model = LamRAQwen2VL(model_args.model_name)
            setattr(base_model, 'config', config)
        elif model_args.model_backbone == LamRA_QWEN2_5:
            base_model = LamRAQwen25VL(model_args.model_name)
            setattr(base_model, 'config', config)
        elif model_args.model_backbone == COLPALI:
            base_model = ColPali.from_pretrained(model_args.model_name)
            setattr(base_model, 'config', config)
        elif model_args.model_backbone == INTERNVL3:
            # Loading the base model
            base_model = backbone2model[model_args.model_backbone].from_pretrained(model_args.model_name)
        else:
            # Loading external base model from HF
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_name_or_path, **kwargs, config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)

        # if model_args.base_lora_checkpoint_path is not None:
        #     print_master(f'Loading SFT checkpoint from {model_args.base_lora_checkpoint_path}')
        #     base_lora_checkpoint_path = model_args.base_lora_checkpoint_path
        #     sft_lora_config = LoraConfig.from_pretrained(base_lora_checkpoint_path)
        #     sft_state_dict = torch.load(os.path.join(base_lora_checkpoint_path, 'adapter_model.bin'), map_location='cpu')
        #     if 'encoder.' in list(sft_state_dict.keys())[0]:
        #         sft_state_dict = {k.replace("encoder.", ""): v for k, v in sft_state_dict.items()}
        #         torch.save(sft_state_dict, os.path.join(base_lora_checkpoint_path, 'adapter_model.bin'))
        #     sft_lora_model = PeftModel.from_pretrained(base_model, base_lora_checkpoint_path, config=sft_lora_config)

        #     base_model = sft_lora_model.merge_and_unload()


        if model_args.meta_queries is not None and model_args.meta_queries > 0:
            processor.tokenizer.add_special_tokens(
                {'additional_special_tokens': [f'<meta_query_{i}>' for i in range(model_args.meta_queries)]})

            base_model.resize_token_embeddings(len(processor.tokenizer))

        # Building the model on top of the base
        if model_args.lora:
            print_master(f'Loading LoRA from {model_name_or_path}')
            lora_config = LoraConfig.from_pretrained(model_name_or_path)
            base_model = PeftModel.from_pretrained(base_model, model_name_or_path, config=lora_config, is_trainable=is_trainable)
            base_model.load_adapter(model_name_or_path, base_model.active_adapter, is_trainable=is_trainable)
            if not is_trainable:
                base_model = base_model.merge_and_unload()

        base_model.model_backbone = model_args.model_backbone
        pooling_module = None
        if model_args.pooling_module not in {'last', 'eos', 'meta_queries'}:
            print(f'Loading attention pooling module')
            pooler_state_dict = torch.load(os.path.join(model_name_or_path, 'pooling_module', 'pytorch_model.bin'), map_location='cpu')
            pooler_state_dict = {k.replace("pooling_module.", ""):v for k, v in pooler_state_dict.items()}

            if model_args.pooling_module == 'attention_pooler':
                pooling_config = AttentionPoolingConfig.from_pretrained(
                    os.path.join(model_name_or_path, 'pooling_module'),
                    trust_remote_code=True
                )
                pooling_module = AttentionPooler(pooling_config)
            elif model_args.pooling_module == 'nv_embed_pooler':
                pooling_config = NVEmbedPoolingConfig.from_pretrained(
                    os.path.join(model_name_or_path, 'pooling_module'),
                    trust_remote_code=True
                )
                pooling_module = NVEmbedPooler(pooling_config)
            elif model_args.pooling_module == 'mlp':
                pooling_config = MLPPoolingConfig.from_pretrained(
                    os.path.join(model_name_or_path, 'pooling_module'),
                    trust_remote_code=True
                )
                pooling_module = MLPPooler(pooling_config)
            elif model_args.pooling_module == 'patched_pooler':
                pooling_config = MultilayerPatchedPoolerConfig.from_pretrained(
                    os.path.join(model_name_or_path, 'pooling_module'),
                    trust_remote_code=True
                )
                pooling_module = MultilayerPatchedPooler(pooling_config)
            elif model_args.pooling_module == 'truncated_self_pooler':
                pooling_config = TruncatedSelfPoolerConfig.from_pretrained(
                    os.path.join(model_name_or_path, 'pooling_module'),
                    trust_remote_code=True
                )
                pooling_module = TruncatedSelfPooler(base_model, pooling_config)
            elif model_args.meta_queries is not None \
                and model_args.meta_queries > 0 \
                and model_args.meta_queries_aggregate_type == "attention_pooler":
                # attention pooling for meta queries with n pooling_n_queries concatted along the latent dim
                pooling_module = AttentionPooler(AttentionPoolingConfig(input_embed_dim=base_model.config.hidden_size,
                                                                        output_embed_dim=base_model.config.hidden_size,
                                                                        n_queries=model_args.pooling_n_queries,
                                                                        n_head=model_args.pooling_n_heads,
                                                                        aggregate="concat"))
            pooling_module.load_state_dict(pooler_state_dict, strict=True)

        
        model = cls(
                encoder=base_model,
                model_config=model_args,
                pooling_module=pooling_module,
                processor=processor
            )

        return model, processor

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, neg: Dict[str, List[Tensor]] = None, *args, **kwargs):

        qry_reps = tgt_reps = neg_reps = None
        if qry is not None:
            qry_reps, qry_res = self.encode_input(qry, do_sft=self.model_config.do_sft_query)

        if tgt is not None:
            tgt_reps, tgt_res = self.encode_input(tgt, do_sft=self.model_config.do_sft_target)

        if neg is not None:
            negs = []
            for i in neg:
                rep, _ = self.encode_input(i)
                negs.append(rep.unsqueeze(1))
            neg_reps = torch.cat(negs, dim=1)

        loss_dict = {}

        if self.model_config.do_cl:
            if qry_reps is None or tgt_reps is None :

                qry_sft_loss = tgt_sft_loss = None
                loss = 0.
                if self.training and kwargs.get('return_loss', False):
                    if self.model_config.do_sft and qry is not None:
                        qry_sft_loss = qry_res.loss
                        loss = loss + qry_sft_loss
                    if self.model_config.do_sft_target and tgt is not None:
                        tgt_sft_loss = tgt_res.loss
                        loss = loss + tgt_sft_loss
                return {"qry_reps": qry_reps, "tgt_reps": tgt_reps, "neg_reps": neg_reps, "qry_sft_loss": qry_sft_loss, "tgt_sft_loss": tgt_sft_loss, "loss": loss}


            # if self.is_ddp:
            #     all_qry_reps = self._dist_gather_tensor(qry_reps)
            #     all_tgt_reps = self._dist_gather_tensor(tgt_reps)
            # else:
            #     all_qry_reps = qry_reps
            #     all_tgt_reps = tgt_reps
            
            # print(f"all_qry_reps: {all_qry_reps.shape}, all_tgt_reps: {all_tgt_reps.shape}")

            # scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
            # scores = scores.view(all_qry_reps.size(0), -1)
            # target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            # target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))
            # cl_loss = self.cross_entropy(scores / self.temperature, target)

            # # if self.is_ddp:
            # #     cl_loss = cl_loss * self.world_size

            # loss_dict['cl_loss'] = cl_loss
            cl_loss = self.loss_fn(x = qry_reps, y = tgt_reps, hard_neg = neg_reps,
                                   x_task_ids = qry['task_id'], y_task_ids = tgt['task_id'], 
                                   temperature = self.temperature, 
                                   inter_task_temperature = self.inter_task_temperature)
            loss_dict['cl_loss'] = cl_loss

        if self.model_config.do_sft_query:
            loss_dict['qry_sft_loss'] = qry_res.loss
        
        if self.model_config.do_sft_target:
            loss_dict['tgt_sft_loss'] = tgt_res.loss

        # if self.is_ddp:
        #     loss = loss * self.world_size
        loss = 0.
        for k, v in loss_dict.items():
            if v is not None:
                loss += v
        loss_dict['loss'] = loss

        return loss_dict

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        if q_reps.dim() == 3 and p_reps.dim() == 3:
            # late interaction [bs, n_queries, d]
            scores = torch.einsum("bnd,csd->bcns", q_reps, p_reps)
            scores = scores.amax(dim=3).sum(dim=2) / q_reps.shape[1] # normalize by n queries
            return scores
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
