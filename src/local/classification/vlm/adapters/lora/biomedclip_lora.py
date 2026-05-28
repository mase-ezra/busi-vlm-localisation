'''
Referred to PEFT and BiomedCLIP-LoRA examples. This file applies LoRA to the BiomedCLIP vision encoder only. 
Note: BiomedCLIP uses a timm ViT trunk (so we target attn.qkv and attn.proj).
Future work: add optional LoRA support for the text encoder.
See: https://github.com/huggingface/peft
See: https://github.com/LightersWang/BiomedCLIP-LoRA
See: https://github.com/jinggqu/NextGen-UIA
'''

from peft import LoraConfig, get_peft_model
import torch.nn as nn

# Freeze pretrained weights so only the LoRA adapters and our classification linear head are updated during few-shot fine-tuning.
def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

# Return PEFT target module names for the BiomedCLIP vision LoRA.
def get_target_modules(model, num_layers=None, lora_targets=('qkv', 'proj')):
    if not (hasattr(model, 'visual') and hasattr(model.visual, 'trunk') and hasattr(model.visual.trunk, 'blocks')):
        raise ValueError('vision blocks not found.')
    
    trunk = model.visual.trunk
    blocks = trunk.blocks
    num_blocks = len(blocks)

    # For our experiments we use all transformer blocks by default (using num_layers = None). 
    # But if a layer limit is given adapt only the final N blocks to match the OpenAI CLIP LoRA setup.
    if num_layers is None: 
        layers_to_inject = num_blocks   
    else: 
        layers_to_inject = min(num_layers, num_blocks)

    start_idx = num_blocks - layers_to_inject

    target_modules = []

    for i in range(start_idx, num_blocks):
        block = blocks[i]

        if not hasattr(block, 'attn'):
            raise ValueError(f'block {i} missing attn.')

        if 'qkv' in lora_targets:
            if not hasattr(block.attn, 'qkv'):
                raise ValueError(f'block {i} missing attn.qkv.')

            if not isinstance(block.attn.qkv, nn.Linear):
                raise ValueError(f'block {i} attn.qkv is not nn.Linear.')

            target_modules.append(f'visual.trunk.blocks.{i}.attn.qkv')

        if 'proj' in lora_targets:
            if not hasattr(block.attn, 'proj'):
                raise ValueError(f'block {i} missing attn.proj.')

            if not isinstance(block.attn.proj, nn.Linear):
                raise ValueError(f'block {i} attn.proj is not nn.Linear.')

            target_modules.append(f'visual.trunk.blocks.{i}.attn.proj')

    if not target_modules:
        raise ValueError('no LoRA targets found.')

    return target_modules

# Apply LoRA to the BiomedCLIP vision encoder.
def apply_lora(args, model, num_layers=None, lora_rank=16, lora_alpha=32, lora_dropout=0.1, lora_targets=('qkv', 'proj')):
    freeze_model(model)
    
    if args.encoder == 'vision':
        target_modules = get_target_modules(model=model, num_layers=num_layers, lora_targets=lora_targets)

        # Configure PEFT LoRA adapters for the BiomedCLIP vision attention layers.
        peft_config = LoraConfig(
            r = lora_rank, # A smaller rank leads to fewer trainable parameters and more efficient fine-tuning.
            lora_alpha = lora_alpha, # Used to scale the LoRA update.
            lora_dropout = lora_dropout, # Regularises the adapter during few-shot training.
            target_modules = target_modules, # Our default targets attn.qkv and attn.proj in selected ViT blocks.
            bias='none' # Keeps fine-tuning parameter-efficient.
        )
        
        # Wrap the base model and PEFT configuration with get_peft_model.
        model = get_peft_model(model, peft_config)

        trainable, total = count_trainable_parameters(model)
        
        if trainable == 0:
            raise RuntimeError('LoRA injection produced zero trainable parameters.')
        
        # After PEFT wrapping the frozen BiomedCLIP weights remain frozen and the LoRA weights are trainable.
        model.print_trainable_parameters()

    else: 
        raise ValueError(f'unsupported encoder {args.encoder} for LoRA injection.')
    
    return model, target_modules

def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total