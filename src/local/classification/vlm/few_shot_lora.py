'''
Few-shot LoRA fine-tuning for CLIP variants. This trains LoRA adapters in the vision encoder plus a linear classifier head.
See: https://github.com/JamesQFreeman/LoRA-ViT/
See: https://github.com/LightersWang/BiomedCLIP-LoRA
See: https://github.com/jinggqu/NextGen-UIA/blob/main/src/models/biomedclip/fewshot_classification.py
See: https://github.com/MaxZanella/CLIP-LoRA
'''

from __future__ import annotations
import contextlib
import io
import argparse
import copy
import logging
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix)
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from .adapters.lora.biomedclip_lora import apply_lora as apply_biomedclip_lora
from .adapters.lora.clip_lora import apply_lora as apply_clip_lora
from .helpers import LinearClassifier, load_busi_image_as_pil
from .models import load_biomedclip, load_openai_clip, load_unimedclip

MODEL_NAME_TO_KEY = {
    'openai_clip_vit_b16': 'openai_clip',
    'biomedclip_vit_b16': 'biomedclip',
    'unimedclip': 'unimedclip'
}

def get_args(argv=None):
    parser = argparse.ArgumentParser('few-shot classification using LoRA')

    parser.add_argument(
        '--model_name',
        type=str,
        default='biomedclip_vit_b16',
        choices=['openai_clip_vit_b16', 'biomedclip_vit_b16', 'unimedclip'],
    )

    # Used only for UniMed-CLIP loader.
    parser.add_argument(
        '--project_root',
        type=str,
        default=str(Path.cwd()),
    )

    # Experimental settings.
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--encoder', type=str, default='vision')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=0)

    # LoRA settings.
    parser.add_argument('--lora_layers', type=int, default=None) # Note: lora_layers = None adapts all transformer blocks.
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    # Target attention projections for the different model types.
    parser.add_argument('--clip_lora_targets', nargs='+', default=['q', 'k', 'v', 'o'])
    parser.add_argument('--biomed_lora_targets', nargs='+', default=['qkv', 'proj'])

    # Training settings for the LoRA adapters and classification head.
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)

    # Learning rates.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--head_lr', type=float, default=1e-3)
    parser.add_argument('--lora_lr', type=float, default=1e-4)
    parser.add_argument('--lr_min', type=float, default=1e-7)

    # Optimisation settings.
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--head_weight_decay', type=float, default=0.0)
    parser.add_argument('--lora_weight_decay', type=float, default=0.0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    parser.add_argument('--selection_metric', type=str, default='macro_f1')

    parser.add_argument('--log_train_metrics', action='store_true')
    
    parser.add_argument('--use_eval_preprocess_for_train', action=argparse.BooleanOptionalAction, default=True)
    
    args = parser.parse_args(args=argv)

    args.lora_r = args.lora_rank

    return args

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sample_kshot_indices(labels, k: int, seed: int):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)

    selected = []
    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) < k:
            raise ValueError(f'class {cls} has {len(cls_indices)} samples; need k={k}.')
        chosen = rng.choice(cls_indices, size=k, replace=False)
        selected.extend(chosen.tolist())

    rng.shuffle(selected)
    return np.asarray(selected, dtype=int)

class ImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, preprocess):
        self.df = dataframe.reset_index(drop=True)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = load_busi_image_as_pil(row['image_path'], output_size=224)
        image = self.preprocess(image)
        label = torch.tensor(row['label_index'], dtype=torch.long)
        return image, label

@torch.no_grad()
def infer_feature_dim(clip_model, preprocess, dataframe: pd.DataFrame, device: str):
    image = load_busi_image_as_pil(dataframe.iloc[0]['image_path'], output_size=224)
    image = preprocess(image).unsqueeze(0).to(device)
    return clip_model.encode_image(image).shape[-1]

def encode_logits(clip_model, classifier: nn.Module, images: torch.Tensor):
    features = clip_model.encode_image(images).float()
    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return classifier(features)

def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def get_trainable_parameter_names(model):
    return [name for name, p in model.named_parameters() if p.requires_grad]

def setup_logger(save_dir, name: str, console: bool = False):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    file_handler = logging.FileHandler(save_dir / 'train.log', mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

def _model_key(args) -> str:
    return MODEL_NAME_TO_KEY.get(args.model_name, args.model_name)

def _load_model_and_preprocesses(args):
    if args.model_name == 'biomedclip_vit_b16':
        loaded = load_biomedclip(device=args.device)

        if len(loaded) == 4:
            model, preprocess_train, preprocess_val, _ = loaded
        elif len(loaded) == 3:
            model, preprocess, _ = loaded
            preprocess_train = preprocess
            preprocess_val = preprocess
        else:
            raise ValueError('unexpected load_biomedclip return format.')

        return model, preprocess_train, preprocess_val

    if args.model_name == 'openai_clip_vit_b16':
        model, preprocess, _ = load_openai_clip(model_name='ViT-B/16', device=args.device)
        return model, preprocess, preprocess

    if args.model_name == 'unimedclip':
        project_root = Path(args.project_root) if args.project_root is not None else None

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model, preprocess, _ = load_unimedclip(device=args.device, project_root=project_root)

        return model, preprocess, preprocess

    raise ValueError(f'unsupported model_name: {args.model_name}')

# Apply LoRA to the vision encoder and return the patched model.
def _apply_vision_lora(args, model, show_adapter_log: bool = False):
    def _apply():
        if args.model_name == 'biomedclip_vit_b16':
            patched_model, _ = apply_biomedclip_lora(
                args = args,
                model = model,
                num_layers = args.lora_layers,
                lora_rank = args.lora_rank,
                lora_alpha = args.lora_alpha,
                lora_dropout = args.lora_dropout,
                lora_targets = tuple(args.biomed_lora_targets),
            )
            return patched_model

        if args.model_name in ('openai_clip_vit_b16', 'unimedclip'):
            patched_model, _ = apply_clip_lora(
                model = model,
                lora_r = args.lora_rank,
                lora_alpha = args.lora_alpha,
                lora_dropout = args.lora_dropout,
                num_layers = args.lora_layers,
                enable_lora = tuple(args.clip_lora_targets),
            )
            return patched_model

        raise ValueError(f'unsupported model_name for LoRA: {args.model_name}')

    if show_adapter_log:
        return _apply()

    with contextlib.redirect_stdout(io.StringIO()):
        return _apply()

def prepare_model(args, support_df: pd.DataFrame, logger: logging.Logger | None = None, log_model_summary: bool = False):
    if logger is None:
        logger = logging.getLogger('fewshot_lora')

    clip_model, preprocess_train, preprocess_val = _load_model_and_preprocesses(args)

    if getattr(args, 'use_eval_preprocess_for_train', False):
        preprocess_train = preprocess_val

    clip_model = _apply_vision_lora(args, clip_model, show_adapter_log=log_model_summary)
    clip_model = clip_model.to(args.device).float()

    feature_dim = infer_feature_dim(
        clip_model=clip_model,
        preprocess=preprocess_val,
        dataframe=support_df,
        device=args.device,
    )

    classifier = LinearClassifier(feature_dim, args.num_classes).to(args.device)
    for p in classifier.parameters():
        p.requires_grad = True

    trainable_names = get_trainable_parameter_names(clip_model)
    clip_trainable, clip_total = count_trainable_parameters(clip_model)
    cls_trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

    if log_model_summary:
        summary_message = (
            f'\nLoRA model check: {_model_key(args)}\n'
            f'  trainable vision tensors: {len(trainable_names)}\n'
            f'  vision trainable params: {clip_trainable:,} / {clip_total:,}\n'
            f'  classifier trainable params: {cls_trainable:,}\n'
            f'  total trainable params: {clip_trainable + cls_trainable:,}\n'
        )

        print(summary_message)

        logger.info(summary_message)

    return clip_model, classifier, preprocess_train, preprocess_val, trainable_names

def compute_metrics(labels, preds, probs, class_names=None):
    out = {
        'accuracy': accuracy_score(labels, preds),
        'balanced_accuracy': balanced_accuracy_score(labels, preds),
        'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
        'recall_macro': recall_score(labels, preds, average='macro', zero_division=0),
        'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
        'weighted_f1': f1_score(labels, preds, average='weighted', zero_division=0),
    }

    per_class_precision = precision_score(labels, preds, average=None, zero_division=0)
    per_class_recall = recall_score(labels, preds, average=None, zero_division=0)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)

    for i in range(probs.shape[1]):
        name = class_names[i] if class_names is not None else f'class_{i}'
        out[f'precision_{name}'] = per_class_precision[i]
        out[f'recall_{name}'] = per_class_recall[i]
        out[f'f1_{name}'] = per_class_f1[i]

    try:
        if probs.shape[1] == 2:
            out['auc'] = roc_auc_score(labels, probs[:, 1])
        else:
            out['auc'] = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except ValueError:
        out['auc'] = np.nan

    return out

@torch.no_grad()
def evaluate(clip_model, classifier, loader, device: str, class_names=None):
    clip_model.eval()
    classifier.eval()

    labels_all, preds_all, probs_all = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = encode_logits(clip_model, classifier, images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()

        labels_all.append(labels.cpu().numpy())
        preds_all.append(preds)
        probs_all.append(probs.cpu().numpy())

    labels = np.concatenate(labels_all)
    preds = np.concatenate(preds_all)
    probs = np.concatenate(probs_all)

    metrics = compute_metrics(labels, preds, probs, class_names=class_names)
    return metrics, preds, probs

def make_loaders(
    support_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocess_train,
    preprocess_val,
    batch_size: int,
    num_workers: int,
):
    pin_memory = torch.cuda.is_available()
    return (
        DataLoader(
            ImageDataset(support_df, preprocess_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        DataLoader(
            ImageDataset(val_df, preprocess_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        DataLoader(
            ImageDataset(test_df, preprocess_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    )

def save_outputs(
    save_dir,
    model_key: str,
    k: int,
    seed: int,
    clip_model,
    classifier,
    support_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preds,
    probs,
    class_names,
    result: dict,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            'clip_model': clip_model.state_dict(),
            'classifier': classifier.state_dict(),
            'result': result,
        },
        save_dir / f'{model_key}_lora_k{k}_seed{seed}_best.pth',
    )

    support_df.to_csv(save_dir / f'{model_key}_lora_k{k}_seed{seed}_support.csv', index=False)

    pred_df = test_df.reset_index(drop=True).copy()
    pred_df['pred_label_index'] = preds
    pred_df['correct'] = pred_df['pred_label_index'] == pred_df['label_index']

    for i, name in enumerate(class_names):
        pred_df[f'prob_{name}'] = probs[:, i]

    pred_df.to_csv(save_dir / f'{model_key}_lora_k{k}_seed{seed}_predictions.csv', index=False)

def train_one_kshot(args, train_df, val_df, test_df, class_names, k, seed, support_idx=None, save_dir=None, logger=None, writer=None, log_model_summary=False):
    set_seed(seed)

    if logger is None:
        logger = logging.getLogger('fewshot_lora')

    if support_idx is None:
        support_idx = sample_kshot_indices(train_df['label_index'].values, k=k, seed=seed)
    else:
        support_idx = np.asarray(support_idx, dtype=int)

    support_df = train_df.iloc[support_idx].reset_index(drop=True)

    clip_model, classifier, preprocess_train, preprocess_val, _ = prepare_model(args, support_df, logger=logger, log_model_summary=log_model_summary)
    train_loader, val_loader, test_loader = make_loaders(
        support_df,
        val_df,
        test_df,
        preprocess_train,
        preprocess_val,
        args.batch_size,
        args.num_workers,
    )

    criterion = nn.CrossEntropyLoss()

    lora_params = [p for p in clip_model.parameters() if p.requires_grad]
    head_params = [p for p in classifier.parameters() if p.requires_grad]

    params = lora_params + head_params

    optimizer = torch.optim.AdamW(
        [
            {
                'params': head_params,
                'lr': args.head_lr,
                'weight_decay': args.head_weight_decay,
            },
            {
                'params': lora_params,
                'lr': args.lora_lr,
                'weight_decay': args.lora_weight_decay,
            },
        ],
        betas=(args.beta1, args.beta2),
    )

    effective_accum = max(1, min(args.accumulation_steps, len(train_loader)))
    updates_per_epoch = max(1, int(np.ceil(len(train_loader) / effective_accum)))
    total_updates = max(1, updates_per_epoch * args.epochs)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_updates,
        eta_min=args.lr_min,
    )

    best_score = -np.inf
    best_clip_state = copy.deepcopy(clip_model.state_dict())
    best_cls_state = copy.deepcopy(classifier.state_dict())
    best_epoch = 0
    patience_count = 0

    progress = tqdm(
        range(args.epochs),
        desc=f'{_model_key(args)} k={k} seed={seed}',
        leave=True,
        disable=not getattr(args, 'show_progress', True),
    )

    for epoch in progress:
        clip_model.train()
        classifier.train()
        optimizer.zero_grad(set_to_none=True)

        loss_sum = 0.0

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            loss = criterion(encode_logits(clip_model, classifier, images), labels)
            (loss / effective_accum).backward()
            loss_sum += float(loss.item())

            do_step = ((step + 1) % effective_accum == 0) or ((step + 1) == len(train_loader))

            if do_step:
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        val_metrics, _, _ = evaluate(clip_model, classifier, val_loader, args.device)

        train_metrics = None

        if getattr(args, 'log_train_metrics', False):
            train_metrics, _, _ = evaluate(clip_model, classifier, train_loader, args.device)

        score = float(val_metrics[args.selection_metric])

        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience_count = 0
            best_clip_state = copy.deepcopy(clip_model.state_dict())
            best_cls_state = copy.deepcopy(classifier.state_dict())
        else:
            patience_count += 1

        train_f1_text = ''

        if train_metrics is not None:
            train_f1_text = f"train_f1={train_metrics['macro_f1']:.4f} "

        epoch_msg = (
            f"model={_model_key(args)} k={k} seed={seed} epoch={epoch + 1}/{args.epochs} "
            f"loss={loss_sum / max(len(train_loader), 1):.4f} "
            f"{train_f1_text}"
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"head_lr={optimizer.param_groups[0]['lr']:.2e} "
            f"lora_lr={optimizer.param_groups[1]['lr']:.2e}"
        )

        postfix = {
            'loss': f'{loss_sum / max(len(train_loader), 1):.4f}',
            'val_f1': f'{val_metrics["macro_f1"]:.4f}',
            'best': f'{best_score:.4f}',
            'head_lr': f'{optimizer.param_groups[0]["lr"]:.1e}',
            'lora_lr': f'{optimizer.param_groups[1]["lr"]:.1e}',
        }

        if train_metrics is not None:
            postfix['train_f1'] = f'{train_metrics["macro_f1"]:.4f}'

        progress.set_postfix(**postfix)

        if getattr(args, 'log_epochs', False):
            logger.info(epoch_msg)

        if writer is not None:
            prefix = f'{_model_key(args)}/k{k}/seed{seed}'
            writer.add_scalar(f'{prefix}/train_loss', loss_sum / max(len(train_loader), 1), epoch)

            if train_metrics is not None:
                writer.add_scalar(f'{prefix}/train_accuracy', train_metrics['accuracy'], epoch)
                writer.add_scalar(f'{prefix}/train_macro_f1', train_metrics['macro_f1'], epoch)

            writer.add_scalar(f'{prefix}/val_accuracy', val_metrics['accuracy'], epoch)
            writer.add_scalar(f'{prefix}/val_macro_f1', val_metrics['macro_f1'], epoch)
            writer.add_scalar(f'{prefix}/val_auc', val_metrics['auc'], epoch)
            writer.add_scalar(f'{prefix}/head_lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar(f'{prefix}/lora_lr', optimizer.param_groups[1]['lr'], epoch)
            writer.flush()

        if patience_count >= args.patience:
            break

    clip_model.load_state_dict(best_clip_state)
    classifier.load_state_dict(best_cls_state)

    test_metrics, preds, probs = evaluate(clip_model, classifier, test_loader, args.device)

    done_msg = (
        f"done model={_model_key(args)} k={k} seed={seed} "
        f"best_epoch={best_epoch + 1} "
        f"best_val_f1={best_score:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_f1={test_metrics['macro_f1']:.4f} "
        f"test_auc={test_metrics['auc']:.4f}"
    )
    print(done_msg)
    logger.info(done_msg)

    clip_trainable, clip_total = count_trainable_parameters(clip_model)
    cls_trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

    result = {
        'model': _model_key(args),
        'experiment': 'fewshot_lora',
        'k': k,
        'seed': seed,
        'n_train_samples': len(support_df),
        'best_epoch': int(best_epoch + 1),
        'best_val_macro_f1': best_score,
        'test_accuracy': test_metrics['accuracy'],
        'test_macro_f1': test_metrics['macro_f1'],
        'test_auc': test_metrics['auc'],
        'trainable_params': int(clip_trainable + cls_trainable),
        'total_clip_params': int(clip_total),
        'lora_rank': int(args.lora_rank),
        'lora_alpha': int(args.lora_alpha),
        'lora_dropout': float(args.lora_dropout),
        'lora_layers': None if args.lora_layers is None else int(args.lora_layers),
        'clip_lora_targets': list(getattr(args, 'clip_lora_targets', [])),
        'biomed_lora_targets': list(getattr(args, 'biomed_lora_targets', [])),
        'head_lr': float(args.head_lr),
        'lora_lr': float(args.lora_lr),
        'lr_min': float(args.lr_min),
        'accumulation_steps': int(args.accumulation_steps),
        'use_eval_preprocess_for_train': bool(args.use_eval_preprocess_for_train)
    }

    if save_dir is not None:
        save_outputs(
            save_dir=save_dir,
            model_key=_model_key(args),
            k=k,
            seed=seed,
            clip_model=clip_model,
            classifier=classifier,
            support_df=support_df,
            test_df=test_df,
            preds=preds,
            probs=probs,
            class_names=class_names,
            result=result,
        )

    return result

def run_kshot_experiments(args, train_df, val_df, test_df, class_names, ks, seeds, save_dir=None, support_indices=None,):
    results = []
    model_key = _model_key(args)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(
            save_dir,
            name=f'fewshot_lora_{model_key}',
            console=getattr(args, 'console_log', False),
        )
        writer = SummaryWriter(log_dir=str(save_dir / 'tensorboard'))
    else:
        logger = logging.getLogger(f'fewshot_lora_{model_key}')
        writer = None

    logger.info(f'Starting LoRA few-shot experiments for: {model_key}')
    logger.info(f'k values: {ks}')
    logger.info(f'seeds: {seeds}')
    logger.info(f'LoRA rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}')
    logger.info(f'LoRA layers={args.lora_layers}')
    print(
        f'\nLoRA few-shot: model={model_key} | '
        f'k={ks} | seeds={seeds} | '
        f'rank={args.lora_rank} | alpha={args.lora_alpha} | '
        f'dropout={args.lora_dropout} | layers={args.lora_layers}'
    )

    for k in ks:
        for seed in seeds:
            support_idx = None
            if support_indices is not None:
                support_idx = support_indices.get(int(k), {}).get(int(seed))
                if support_idx is None:
                    raise ValueError(f'no support indices found for k={k} seed={seed}')

            is_first_run = len(results) == 0

            result = train_one_kshot(
                args=args,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                class_names=class_names,
                k=k,
                seed=seed,
                support_idx=support_idx,
                save_dir=save_dir,
                logger=logger,
                writer=writer,
                log_model_summary=is_first_run,
            )
            results.append(result)

    results_df = pd.DataFrame(results)
    summary_df = (
        results_df.groupby('k', as_index=False)
        .agg(
            best_epoch_mean=('best_epoch', 'mean'),
            best_val_macro_f1_mean=('best_val_macro_f1', 'mean'),
            best_val_macro_f1_std=('best_val_macro_f1', 'std'),
            test_accuracy_mean=('test_accuracy', 'mean'),
            test_accuracy_std=('test_accuracy', 'std'),
            test_macro_f1_mean=('test_macro_f1', 'mean'),
            test_macro_f1_std=('test_macro_f1', 'std'),
            test_auc_mean=('test_auc', 'mean'),
            test_auc_std=('test_auc', 'std'),
        )
    )

    if save_dir is not None:
        results_df.to_csv(save_dir / 'results.csv', index=False)
        summary_df.to_csv(save_dir / 'summary.csv', index=False)
        logger.info(f"Saved results to {save_dir / 'results.csv'}")
        logger.info(f"Saved summary to {save_dir / 'summary.csv'}")

    print('\nLoRA summary:')
    print(summary_df.to_string(index=False))
    logger.info('\n' + summary_df.to_string(index=False))

    if writer is not None:
        writer.close()

    return results_df, summary_df