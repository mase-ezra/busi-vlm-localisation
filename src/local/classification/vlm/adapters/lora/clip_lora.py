'''
LoRA adapter for OpenAI CLIP and UniMedCLIP vision encoders. These models use PyTorch nn.MultiheadAttention with packed QKV weights. 
See: https://github.com/KyanChen/MakeMultiHeadNaive/
See: https://github.com/MaxZanella/CLIP-LoRA/blob/main/loralib/easymultiheadattention.py
See: https://github.com/jinggqu/NextGen-UIA/blob/main/src/adapters/lora.py
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Freeze all model parameters.
def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

# Count trainable and total parameters.
def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

# LoRA wrapper for one frozen nn.Linear layer.
class LinearLoRA(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 16, lora_alpha: int = 32, dropout_rate: float = 0.1):
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f'expected nn.Linear, got {type(base_linear)}')

        self.base = base_linear
        self.r = int(r)
        self.scaling = lora_alpha / self.r if self.r > 0 else 1.0
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad = False

        if self.r > 0:
            self.lora_A = nn.Linear(base_linear.in_features, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, base_linear.out_features, bias=False)

            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        output = self.base(x)

        if self.r <= 0:
            return output

        lora_output = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return output + lora_output

# Split PyTorch nn.MultiheadAttention packed QKV weights into separate Linear layers.
def split_multihead_attention(mha: nn.MultiheadAttention):
    embed_dim = mha.embed_dim
    has_bias = mha.in_proj_bias is not None

    q_proj = nn.Linear(embed_dim, embed_dim, bias=has_bias)
    k_proj = nn.Linear(embed_dim, embed_dim, bias=has_bias)
    v_proj = nn.Linear(embed_dim, embed_dim, bias=has_bias)
    proj = nn.Linear(embed_dim, embed_dim, bias=mha.out_proj.bias is not None)

    with torch.no_grad():
        q_proj.weight.copy_(mha.in_proj_weight[:embed_dim])
        k_proj.weight.copy_(mha.in_proj_weight[embed_dim: 2 * embed_dim])
        v_proj.weight.copy_(mha.in_proj_weight[2 * embed_dim:])

        if has_bias:
            q_proj.bias.copy_(mha.in_proj_bias[:embed_dim])
            k_proj.bias.copy_(mha.in_proj_bias[embed_dim: 2 * embed_dim])
            v_proj.bias.copy_(mha.in_proj_bias[2 * embed_dim:])

        proj.weight.copy_(mha.out_proj.weight)

        if proj.bias is not None:
            proj.bias.copy_(mha.out_proj.bias)

    return q_proj, k_proj, v_proj, proj

# LoRA wrapper for OpenAI CLIP / UniMedCLIP-style nn.MultiheadAttention.
class PlainMultiheadAttentionLoRA(nn.Module):
    def __init__(self, existing_mha: nn.MultiheadAttention, enable_lora=('q', 'k', 'v', 'o'), r: int = 16, lora_alpha: int = 32, dropout_rate: float = 0.1):
        super().__init__()

        if not isinstance(existing_mha, nn.MultiheadAttention):
            raise TypeError(f'expected nn.MultiheadAttention, got {type(existing_mha)}')

        self.embed_dim = existing_mha.embed_dim
        self.num_heads = existing_mha.num_heads
        self.head_dim = existing_mha.head_dim
        self.batch_first = existing_mha.batch_first

        self.dropout = 0.0

        q_proj, k_proj, v_proj, proj = split_multihead_attention(existing_mha)

        self.q_proj = (
            LinearLoRA(q_proj, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate)
            if 'q' in enable_lora else q_proj
        )
        self.k_proj = (
            LinearLoRA(k_proj, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate)
            if 'k' in enable_lora else k_proj
        )
        self.v_proj = (
            LinearLoRA(v_proj, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate)
            if 'v' in enable_lora else v_proj
        )
        self.proj = (
            LinearLoRA(proj, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate)
            if 'o' in enable_lora else proj
        )

        self.freeze_non_lora_params()

    def freeze_non_lora_params(self):
        for name, p in self.named_parameters():
            if 'lora_A' not in name and 'lora_B' not in name:
                p.requires_grad = False

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None, average_attn_weights=True, is_causal=False, **kwargs):
        if key_padding_mask is not None:
            raise NotImplementedError('key_padding_mask is not supported in this vision lora wrapper.')

        if need_weights:
            raise NotImplementedError('attention weights are not returned by this vision lora wrapper.')

        is_batched = query.dim() == 3

        if self.batch_first and is_batched:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        dropout_p = self.dropout if self.training else 0.0

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

        attn_output = (attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim))

        attn_output = self.proj(attn_output)

        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, None


# Apply LoRA to OpenAI CLIP and UniMedCLIP vision encoder.
# For a fair comparison with BiomedCLIP, which adapts fused qkv and proj layers, this adapter enables LoRA on q, k, v, and output projection layers.
def apply_lora(model, lora_r=16, lora_alpha=32, lora_dropout=0.1, num_layers=None, enable_lora=('q', 'k', 'v', 'o')):
    freeze_model(model)

    lora_count = 0

    if isinstance(enable_lora, str):
        enable_lora = tuple(enable_lora)

    if not (hasattr(model, 'visual') and hasattr(model.visual, 'transformer') and hasattr(model.visual.transformer, 'resblocks')):
        raise ValueError('expected model.visual.transformer.resblocks for clip-style vision lora.')

    blocks = model.visual.transformer.resblocks
    layers_to_inject = len(blocks) if num_layers is None else min(num_layers, len(blocks))
    start_idx = len(blocks) - layers_to_inject

    for i in range(start_idx, len(blocks)):
        block = blocks[i]

        if not hasattr(block, 'attn'):
            raise ValueError(f'block {i} missing attn.')

        if not isinstance(block.attn, nn.MultiheadAttention):
            raise TypeError(f'block {i}.attn is {type(block.attn)}, not nn.MultiheadAttention. this adapter supports openai clip / unimedclip-style multihead attention.')

        block.attn = PlainMultiheadAttentionLoRA(existing_mha=block.attn, enable_lora=enable_lora, r=lora_r, lora_alpha=lora_alpha, dropout_rate=lora_dropout)

        lora_count += 1

    trainable, total = count_trainable_parameters(model)

    print(f'injected lora adapters to {lora_count} layers (clip vision encoder)')
    print(f'trainable params: {trainable:,} / {total:,}')

    return model, lora_count

# index_positions_model = {
#        'ViT-B/16': {
#         'top': [11],
#         'top3': [9, 10, 11],
#         'bottom': [0, 1, 2, 3],
#         'mid': [4, 5, 6, 7],
#         'up': [8, 9, 10, 11],
#         'half-up': [6, 7, 8, 9, 10, 11],
#         'half-bottom': [0, 1, 2, 3, 4, 5],
#         'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}, 
# }
