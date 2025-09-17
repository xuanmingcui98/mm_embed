from typing import List, Optional

import torch
from torch import nn, Tensor
from torchmultimodal.modules.layers.multi_head_attention import (
    MultiHeadAttentionWithCache,
)
from peft import PeftModel
from transformers import PreTrainedModel, PretrainedConfig
from einops import rearrange, repeat
from copy import deepcopy

class AttentionPoolingConfig(PretrainedConfig):
    model_type = "attention_pooler"
    is_composition = False
    _name_or_path = model_type

    def __init__(
        self,
        input_embed_dim: int = 1536,
        output_embed_dim: int = 1536,
        n_head: int = 8,
        n_queries: int = 1,
        layer_norm_eps: float = 1e-5,
        aggregate: str = "mean",
        **kwargs,
    ):
        self.input_embed_dim = input_embed_dim
        self.output_embed_dim = output_embed_dim
        self.n_head = n_head
        self.n_queries = n_queries
        self.layer_norm_eps = layer_norm_eps
        self.aggregate = aggregate
        
        super().__init__(**kwargs)

class AttentionPooler(PreTrainedModel):
    """
    Attention pooling layer: pools inputs to sequence length n_queries by performing
        cross-attention with learned query embeddings. Based on the CoCa implementation
        in open_clip repo: https://fburl.com/3mlrd0ir.
    Args:
        input_embed_dim (int): Embedding dimension of inputs.
        output_embed_dim (int): Embedding dimension of outputs.
        n_head (int): Number of attention heads.
        n_queries (int): Number of queries. Defaults to 256
        layer_norm_eps (Optional[float]): Epsilon for layer norms. Defaults to 1e-5
    """
    config_class = AttentionPoolingConfig

    def __init__(
        self,
        config: AttentionPoolingConfig,
    ):
        super().__init__(config)
        self.config = config
        self.query = nn.Parameter(torch.randn(config.n_queries, config.output_embed_dim))
        self.attn = MultiHeadAttentionWithCache(
            dim_q=config.output_embed_dim, dim_kv=config.input_embed_dim, num_heads=config.n_head
        )
        self.ln_q = nn.LayerNorm(config.output_embed_dim, config.layer_norm_eps)
        self.ln_k = nn.LayerNorm(config.input_embed_dim, config.layer_norm_eps)
        self.ln_post = nn.LayerNorm(config.output_embed_dim, config.layer_norm_eps)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Inputs:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_embed_dim).
            padding_mask (optional Tensor): Attention mask of shape bsz x 1 x 1 x seq_len
                (for broadcasting to num_heads and target_len dimension).
        Returns:
            Attention pooled tensor with shape
                (batch_size, n_queries, output_embed_dim).
        """
        x = self.ln_k(x)
        query = self.ln_q(self.query)
        batch_size = x.shape[0]

        # (n_queries, output_embed_dim) -> (batch_size, n_queries, output_embed_dim)
        query = self._repeat(query, batch_size)

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1).bool()

        out = self.attn(query, x, x, attn_mask=padding_mask)
        assert isinstance(out, Tensor)
        out = self.ln_post(out)

        if self.config.aggregate == "mean":
            out = out.mean(1)
        elif self.config.aggregate == "concat":
            out = rearrange(out, "b n d -> b (n d)")
        return out

    def _repeat(self, query: Tensor, N: int) -> Tensor:
        return query.unsqueeze(0).repeat(N, 1, 1)


class NVEmbedPoolingConfig(PretrainedConfig):
    model_type = "nvembed_pooler"
    is_composition = False
    _name_or_path = model_type

    def __init__(
        self,
        num_latents_value: int=512,
        num_cross_heads: int=8,
        # output_normalize: bool=True,
        hidden_dim: int=4096,
        latent_dim: int=4096,
        cross_dim_head: int=4096,
        **kwargs,
    ):
        self.num_latents_value = num_latents_value
        self.num_cross_heads = num_cross_heads
        # self.output_normalize = output_normalize
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cross_dim_head = cross_dim_head

        super().__init__(**kwargs)

class NVEmbedPreNorm(torch.nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.LayerNorm(dim)
        self.norm_context = torch.nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)
        return self.fn(x, **kwargs)

class NVEmbedGEGLU(torch.nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * torch.nn.functional.gelu(gates)

class NVEmbedFeedForward(torch.nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, dim * mult * 2),
            NVEmbedGEGLU(),
            torch.nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)
    
def default(val, d):
    return val if exists(val) else d

def exists(val):
    return val is not None

class NVEmbedAttention(torch.nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = torch.nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = torch.nn.Linear(inner_dim, query_dim, bias = False)

    def forward(self, x, context = None, mask = None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)


class NVEmbedPooler(PreTrainedModel):
    config_class = NVEmbedPoolingConfig

    def __init__(self, config: NVEmbedPoolingConfig):
        super().__init__(config)
        ## cross-attention block
        num_latents, latent_dim, cross_heads, cross_dim_head = config.num_latents_value, config.latent_dim, config.num_cross_heads, config.cross_dim_head
        dim = config.hidden_dim
        # init latent_attention and latents
        self.cross_attend_blocks = torch.nn.ModuleList([
            NVEmbedPreNorm(latent_dim, NVEmbedAttention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head),
                    context_dim = dim),
            NVEmbedPreNorm(latent_dim, NVEmbedFeedForward(latent_dim)),
        ])
        # self.output_normalize = config.output_normalize
        self.register_parameter("latents", torch.nn.Parameter(torch.randn(num_latents, latent_dim)))

        # # initialize all parameters with zero 
        # for name, param in self.named_parameters():
        #     torch.nn.init.zeros_(param)

    def forward(self, hiddens, attention_mask: torch.Tensor=None):
        ## cross-attention block
        cross_attn, cross_ff = self.cross_attend_blocks
        b, *_, device = *hiddens.shape, hiddens.device
        x = repeat(self.latents, 'n d -> b n d', b = b)
        hiddens = cross_attn(hiddens, context = x, mask = None) + hiddens
        hiddens = cross_ff(hiddens) + hiddens
        if attention_mask !=None:
            s = torch.sum(hiddens * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            hiddens = s / d
            # if self.output_normalize:
            #     hiddens = torch.nn.functional.normalize(hiddens, p=2, dim=-1)
        return hiddens


class MLPPoolingConfig(PretrainedConfig):
    model_type = "mlp_pooler"
    is_composition = False
    _name_or_path = model_type

    def __init__(
        self,
        input_embed_dim: int = 1536,
        output_embed_dim: int = 1536,
        dropout: float = 0.1,
        **kwargs,
    ):
        self.input_embed_dim = input_embed_dim
        self.output_embed_dim = output_embed_dim
        self.dropout = dropout
        
        super().__init__(**kwargs)

class MLPPooler(PreTrainedModel):
    config_class = MLPPoolingConfig
    def __init__(self, config):
        super().__init__(config)
        self.fc1 = nn.Linear(config.input_embed_dim, config.input_embed_dim * 4)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.input_embed_dim * 4, config.output_embed_dim)
    def forward(self, x):
        residual = x
        x = self.act(self.fc1(x))
        x = self.fc2(self.dropout(x)) + residual
        return x


class MultilayerPatchedPoolerConfig(PretrainedConfig):
    model_type = "multilayer_patched_pooler"
    is_composition = False
    _name_or_path = model_type

    def __init__(
            self,
            input_embed_dim: int = 1536,
            output_embed_dim: int = 1536,
            n_head: int = 8,
            n_queries: int = 64,
            layer_norm_eps: float = 1e-5,
            num_layers: int = 8,
            last_n_layers: int = None,
            dropout: float = 0.1,
            **kwargs):
        
        self.input_embed_dim = input_embed_dim
        self.output_embed_dim = output_embed_dim
        self.n_head = n_head
        self.n_queries = n_queries
        self.layer_norm_eps = layer_norm_eps
        self.num_layers = num_layers
        self.aggregate = None
        self.last_n_layers = last_n_layers
        self.dropout = dropout

        super().__init__(**kwargs)

class PatchedPoolerAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.input_embed_dim, eps=config.layer_norm_eps)

        self.self_attn = MultiHeadAttentionWithCache(
            dim_q=config.input_embed_dim,
            dim_kv=config.input_embed_dim,
            num_heads=config.n_head,
        )
        self.ln2 = nn.LayerNorm(config.input_embed_dim, eps=config.layer_norm_eps)
        self.cross_attn = AttentionPooler(config)
        self.ln3 = nn.LayerNorm(config.input_embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLPPooler(config)
    
    def forward(self, x, cross, padding_mask=None):

        x = self.ln1(x)

        residual = x
        x = self.cross_attn(cross, padding_mask=padding_mask)
        x = x + residual

        x = self.ln2(x)

        residual = x
        x = self.self_attn(x, x, x)
        x = x + residual

        x = self.ln3(x)

        x = self.mlp(x)
        
        return x

class GEGLU(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate_proj = torch.nn.Linear(config.input_embed_dim)
    
    def forward(self, x):
        gates = self.gate_proj(x)
        return x * torch.nn.functional.gelu(gates)

class MultilayerPatchedPooler(PreTrainedModel):
    """
    Multilayer Patched Pooler: pools inputs to sequence length n_queries by performing
        cross-attention with learned query embeddings. Based on the CoCa implementation
        in open_clip repo: https://fburl.com/3mlrd0ir.
    Args:
        input_embed_dim (int): Embedding dimension of inputs.
        output_embed_dim (int): Embedding dimension of outputs.
        n_head (int): Number of attention heads.
        n_queries (int): Number of queries. Defaults to 256
        layer_norm_eps (Optional[float]): Epsilon for layer norms. Defaults to 1e-5
    """
    config_class = MultilayerPatchedPoolerConfig

    def __init__(self, config: MultilayerPatchedPoolerConfig):
        super().__init__(config)
        self.config = config


        self.queries = nn.Parameter(torch.zeros(1, config.n_queries, config.input_embed_dim))
        self.ln = nn.LayerNorm(config.input_embed_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.attention_blocks = nn.ModuleList([
            PatchedPoolerAttentionBlock(config) for _ in range(config.num_layers)
        ])

        # self.gates = nn.ModuleList([
        #     GEGLU(config) for _ in range(config.num_layers)
        # ])
        self.last_n_layers = config.last_n_layers

    def forward(self, hidden_states: List[Tensor], padding_mask: Optional[Tensor] = None) -> Tensor:
        
        n_original_layers = len(hidden_states) - 1

        if self.last_n_layers is not None:
            if self.last_n_layers > n_original_layers:
                raise ValueError(f"last_n ({self.last_n_layers}) cannot be greater than the number of original layers ({n_original_layers}).")
            # for hidden_states with length N (original LLM's number of layers), we take the last N - last_n layers
            hidden_states = hidden_states[-self.last_n_layers:]
            n_original_layers = self.last_n_layers

        n = n_original_layers // self.config.num_layers
        # for hidden_states with length N (original LLM's number of layers), we take every N // num_layers layers, including the last layer
        hidden_states = [hidden_states[min(i + n - 1, n_original_layers - 1)] for i in range(0, n_original_layers, n)]
        
        assert len(hidden_states) == self.config.num_layers, \
            f"Expected {self.config.num_layers} layers, but got {len(hidden_states)} layers."


        hidden_state = self.dropout(self.ln(self.queries))
        for block, cross in zip(self.attention_blocks, hidden_states):

            # hidden_state = gate(hidden_state)
            residual = hidden_state
            hidden_state = block(hidden_state, cross, padding_mask=padding_mask)
            hidden_state = hidden_state + residual

            residual = hidden_state
        
        return hidden_state.mean(dim=1)


class TruncatedSelfPoolerConfig(PretrainedConfig):
    model_type = "truncated_self_pooler"
    is_composition = False
    _name_or_path = model_type

    def __init__(
            self,
            num_layers: int = 8,
            last_n_layers: int = None,
            aggregation: str = "last",
        **kwargs):
        self.num_layers = num_layers
        self.last_n_layers = last_n_layers
        self.aggregation = aggregation
        super().__init__(**kwargs)

class TruncatedSelfPooler(PreTrainedModel):
    config_class = TruncatedSelfPoolerConfig

    def __init__(self, model, config: TruncatedSelfPoolerConfig):
        super().__init__(config)

        self.last_n_layers     = config.last_n_layers
        self.num_pooler_layers = config.num_layers
        self.aggregation       = config.aggregation

        
        if isinstance(model, PeftModel):
            self._base_model = [model.base_model.model]
            layers = model.base_model.model.model.layers
        else:
            self._base_model = [model]
            layers = model.model.layers
        total_layers = len(layers)
        if self.last_n_layers is not None:
            start_idx = total_layers - self.last_n_layers
        else:
            start_idx = 0

        orig_layers = layers[start_idx:]
        n_orig      = len(orig_layers)
        stride      = max(1, n_orig // self.num_pooler_layers)

        picked = [min(i + stride - 1, n_orig - 1) for i in range(0, n_orig, stride)]
        picked = picked[: self.num_pooler_layers]
        self.picked = [x + start_idx for x in picked]

        self.pooler_layers = nn.ModuleList([
            deepcopy(orig_layers[idx]) for idx in picked
        ])

        # hf return hidden states, starting at input_embeds, so it returns n_layers + 1 hidden states
        # meaning we don't need to add 1 to the index
        self.input_hidden_state_ind = start_idx + picked[0]

    @property
    def base_model(self):
        return self._base_model[0]

    def forward(self, hidden_states: list, padding_mask=None):
        # hidden_states[0] = embeddings, hidden_states[1] = after layer 0, … 
        hs = hidden_states[self.input_hidden_state_ind]


        if "qwen2" == "qwen2":  # dummy condition. Currently only support qwen2 vl series
            position_embeddings = self.base_model.model.current_position_embeddings

        for layer in self.pooler_layers:
            # make sure you’re calling the right signature here;
            # some models expect (hs, attention_mask=…) instead.
            hs = layer(hs, attention_mask=padding_mask, position_embeddings=position_embeddings)[0]

        if self.aggregation == "mean":
            return hs.mean(dim=1)
        elif self.aggregation == "last":
            return hs[:, -1, :]
        else:
            raise ValueError(f"Unknown aggregation {self.aggregation}")
        

