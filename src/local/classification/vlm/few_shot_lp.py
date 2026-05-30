'''
Few-shot linear probe training for CLIP variants. We train a linear classifier on top of frozen CLIP image encoders with varying amounts of training data to evaluate data efficiency.
See: https://github.com/FereshteShakeri/FewShot-CLIP-Strong-Baseline
See: https://github.com/batmanlab/Mammo-CLIP/
See: https://github.com/jinggqu/NextGen-UIA/
'''

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from .helpers import LinearClassifier

@dataclass
class LinearProbeConfig:
    '''Settings for the sklearn linear probe baseline.'''
    max_iter: int = 5000
    class_weight: str = 'balanced'
    c_values: tuple = (0.01, 0.1, 1.0, 10.0, 100.0)
    selection_metric: str = 'macro_f1'

def setup_logger(save_dir, name='fewshot_linear_probe'):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    file_handler = logging.FileHandler(save_dir / 'train.log', mode='a')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def set_seed(seed):
    '''Set random seeds for repeatable few-shot sampling and training.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_tensor(x, dtype=None):
    '''Convert arrays or tensors to a torch tensor.'''
    out = x.detach() if torch.is_tensor(x) else torch.as_tensor(x)
    return out.to(dtype) if dtype is not None else out

def to_numpy(x):
    '''Convert tensor or array to numpy.'''
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()

    return np.asarray(x)

class FeatureDataset(Dataset):
    '''Dataset wrapper for pre-extracted image features.'''

    def __init__(self, features, labels):
        self.features = to_tensor(features, torch.float32)
        self.labels = to_tensor(labels, torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def is_shots_per_class(value):
    '''Return True for fixed shots-per-class values, False for ratio fractions.'''
    return (
        isinstance(value, int)
        or (isinstance(value, float) and value.is_integer() and value >= 1.0)
    )


def sample_ratio_indices(labels, ratio, seed, num_classes, min_per_class=1):
    '''Stratified few-shot sampling from the training labels.

    If ratio is a fraction <= 1.0, we sample that fraction from each class.
    If ratio is an integer or whole number >= 1, we sample that many examples per class.
    '''
    labels = to_tensor(labels, torch.long).cpu().numpy()
    rng = np.random.default_rng(seed)
    sampled_indices = []

    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0]

        if len(class_indices) == 0:
            raise ValueError(f'no samples found for class index {class_idx}')

        if is_shots_per_class(ratio):
            n_samples = int(round(ratio))
        else:
            n_samples = int(round(len(class_indices) * ratio))

        n_samples = max(min_per_class, min(n_samples, len(class_indices)))

        chosen = rng.choice(class_indices, size=n_samples, replace=False)
        sampled_indices.extend(chosen.tolist())

    sampled_indices = np.array(sampled_indices)
    rng.shuffle(sampled_indices)

    return sampled_indices

def make_kshot_indices(dataframe, label_col, shots_per_class, seeds):
    '''
    Create shared class-balanced k-shot support indices.
    '''

    labels = dataframe[label_col].values
    num_classes = len(np.unique(labels))

    indices = {}

    for shot in shots_per_class:
        indices[int(shot)] = {}

        for seed in seeds:
            indices[int(shot)][int(seed)] = sample_ratio_indices(
                labels=labels,
                ratio=int(shot),
                seed=int(seed),
                num_classes=num_classes,
                min_per_class=int(shot),
            )

    return indices

def make_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps):
    '''Linear warmup followed by cosine decay.'''

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def compute_metrics(y_true, y_pred, probs, class_names):
    '''Compute overall and per-class classification metrics.'''
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }

    try:
        metrics['auc'] = roc_auc_score(
            y_true,
            probs,
            multi_class='ovr',
            average='macro'
        )
    except ValueError:
        metrics['auc'] = np.nan

    labels = np.arange(len(class_names))

    per_class_precision = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0
    )

    per_class_recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0
    )

    per_class_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0
    )

    for i, class_name in enumerate(class_names):
        metrics[f'{class_name}_precision'] = per_class_precision[i]
        metrics[f'{class_name}_recall'] = per_class_recall[i]
        metrics[f'{class_name}_f1'] = per_class_f1[i]

    return metrics

@torch.no_grad()
def predict_linear_probe(model, features, labels, class_names, device='cuda', batch_size=64):
    '''Run logistic-regression inference and return metrics, predictions, and probabilities.'''
    x = to_numpy(features).astype(np.float32)
    y = to_numpy(labels).astype(int)

    probs = model.predict_proba(x)
    preds = model.predict(x)

    metrics = compute_metrics(
        y_true=y,
        y_pred=preds,
        probs=probs,
        class_names=class_names
    )

    return metrics, preds, probs

def train_linear_probe(
    train_features,
    train_labels,
    val_features,
    val_labels,
    class_names,
    device='cuda',
    config=None,
    verbose=False
):
    '''Train logistic regression on frozen image features and select C on validation.'''
    if config is None:
        config = LinearProbeConfig()

    x_train = to_numpy(train_features).astype(np.float32)
    y_train = to_numpy(train_labels).astype(int)

    x_val = to_numpy(val_features).astype(np.float32)
    y_val = to_numpy(val_labels).astype(int)

    best_model = None
    best_score = -np.inf
    best_c = None

    for c_value in config.c_values:
        model = LogisticRegression(
            C=c_value,
            max_iter=config.max_iter,
            class_weight=config.class_weight,
            solver='lbfgs',
            random_state=0,
        )

        model.fit(x_train, y_train)

        val_probs = model.predict_proba(x_val)
        val_preds = model.predict(x_val)

        val_metrics = compute_metrics(
            y_true=y_val,
            y_pred=val_preds,
            probs=val_probs,
            class_names=class_names,
        )

        score = val_metrics[config.selection_metric]

        if score > best_score:
            best_score = score
            best_model = model
            best_c = c_value

    if verbose:
        print(f'best c={best_c} val_{config.selection_metric}={best_score:.4f}')

    return best_model, {
        'best_c': best_c,
        'best_val_score': best_score,
    }

def run_linear_probe_once(model_name, train_features, train_labels, val_features, val_labels, test_features, test_labels, class_names, ratio, seed, device='cuda', config=None, log_dir=None, verbose=False, support_indices=None,):
    '''Run one linear-probe experiment for one train ratio and seed.'''
    set_seed(seed)

    train_labels_tensor = to_numpy(train_labels).astype(int)

    if support_indices is None:
        num_classes = len(class_names)
        support_indices = sample_ratio_indices(
            labels=train_labels_tensor,
            ratio=ratio,
            seed=seed,
            num_classes=num_classes,
            min_per_class=1,
        )
    else:
        support_indices = np.asarray(support_indices, dtype=int)

    support_features = to_numpy(train_features).astype(np.float32)[support_indices]
    support_labels = train_labels_tensor[support_indices]

    model, best_info = train_linear_probe(
        train_features=support_features,
        train_labels=support_labels,
        val_features=val_features,
        val_labels=val_labels,
        class_names=class_names,
        device=device,
        config=config,
        verbose=verbose,
    )

    test_metrics, _, _ = predict_linear_probe(
        model=model,
        features=test_features,
        labels=test_labels,
        class_names=class_names,
        device=device,
    )

    is_shots = is_shots_per_class(ratio)

    test_metrics.update(
        {
            'run_id': f"{model_name}__linear_probe__{'shots' if is_shots else 'ratio'}{int(ratio) if is_shots else ratio:g}__seed{seed}",
            'model': model_name,
            'experiment': 'fewshot_linear_probe',
            'train_ratio': None if is_shots else ratio,
            'train_ratio_percent': None if is_shots else ratio * 100,
            'shots_per_class': int(ratio) if is_shots else None,
            'seed': seed,
            'n_train_samples': len(support_indices),
            'best_c': best_info['best_c'],
            'best_val_score': best_info['best_val_score'],
            'best_val_macro_f1': best_info['best_val_score'],
            'test_accuracy': test_metrics['accuracy'],
            'test_balanced_accuracy': test_metrics['balanced_accuracy'],
            'test_macro_f1': test_metrics['macro_f1'],
            'test_weighted_f1': test_metrics['weighted_f1'],
            'test_auc': test_metrics['auc'],
        }
    )

    if log_dir is not None:
        run_dir = Path(log_dir) / model_name / f"{'shots' if is_shots else 'ratio'}{int(ratio) if is_shots else ratio:g}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(run_dir))
        writer.add_scalar('test/accuracy', test_metrics['accuracy'], 0)
        writer.add_scalar('test/balanced_accuracy', test_metrics['balanced_accuracy'], 0)
        writer.add_scalar('test/macro_f1', test_metrics['macro_f1'], 0)
        writer.add_scalar('test/auc', test_metrics.get('auc', float('nan')), 0)
        writer.add_scalar('validation/best_macro_f1', best_info['best_val_score'], 0)
        writer.close()

    return test_metrics

# Run the linear-probe experiments for all k-shots and seeds.
def run_linear_probe_experiments(model_name, train_features, train_labels, val_features, val_labels, test_features, test_labels, class_names, ratios, seeds, device='cuda', config=None, log_dir=None, verbose=False, kshot_indices=None):
    if config is None:
        config = LinearProbeConfig()

    logger = None
    output_dir = None
    if log_dir is not None:
        output_dir = Path(log_dir) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(output_dir, name=f'fewshot_linear_probe_{model_name}')
        logger.info(f'Starting linear probe few-shot experiments for: {model_name}')
        logger.info(f'ratios/shots: {ratios}')
        logger.info(f'seeds: {seeds}')
        logger.info(f'C values: {config.c_values}')

    all_results = []

    for ratio in ratios:
        ratio_results = []

        for seed in seeds:
            support_indices = None

            if kshot_indices is not None:
                support_indices = kshot_indices.get(int(ratio), {}).get(int(seed))

                if support_indices is None:
                    raise ValueError(f'no support indices found for ratio={ratio} seed={seed}')
                
            metrics = run_linear_probe_once(model_name=model_name, train_features=train_features, train_labels=train_labels, val_features=val_features, val_labels=val_labels, test_features=test_features, test_labels=test_labels, class_names=class_names, ratio=ratio, seed=seed, device=device, config=config, log_dir=log_dir, verbose=verbose, support_indices=support_indices)
            all_results.append(metrics)
            ratio_results.append(metrics)
            is_shots = is_shots_per_class(ratio)
            label = f'shots={int(ratio)}' if is_shots else f'ratio={ratio:.2f}'
            
            message = (
                f"model={model_name} {label} seed={seed} "
                f"| n={metrics['n_train_samples']} "
                f"| c={metrics['best_c']} "
                f"| val_f1={metrics['best_val_macro_f1']:.4f} "
                f"| test_acc={metrics['test_accuracy']:.4f} "
                f"| test_f1={metrics['test_macro_f1']:.4f} "
                f"| test_auc={metrics['test_auc']:.4f}"
            )

            if verbose:
                print(message)

            if logger is not None:
                logger.info(message)

        # I've just made the k-shot logging clearer.
        ratio_df = pd.DataFrame(ratio_results)
        is_shots = is_shots_per_class(ratio)
        label = f'k={int(ratio)}' if is_shots else f'ratio={ratio:.2f}'

        summary_message = (
            f"model={model_name} {label} | seeds={len(ratio_results)} "
            f"| acc={ratio_df['test_accuracy'].mean():.4f}+/-{ratio_df['test_accuracy'].std():.4f} "
            f"| f1={ratio_df['test_macro_f1'].mean():.4f}+/-{ratio_df['test_macro_f1'].std():.4f} "
            f"| auc={ratio_df['test_auc'].mean():.4f}+/-{ratio_df['test_auc'].std():.4f}"
        )

        print(summary_message)

        if logger is not None:
            logger.info(summary_message)

    results_df = pd.DataFrame(all_results)
    results_df['k'] = results_df['shots_per_class']

    agg_metrics = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'auc', 'n_train_samples']

    agg_dict = {f'{m}_mean': (m, 'mean') for m in agg_metrics}
    agg_dict.update({f'{m}_std': (m, 'std') for m in agg_metrics})

    aggregate_df = (
        results_df
        .groupby(
            ['model', 'experiment', 'train_ratio', 'train_ratio_percent', 'shots_per_class'],
            as_index=False,
            dropna=False,
        )
        .agg(**agg_dict)
        .sort_values(['model', 'shots_per_class', 'train_ratio'], na_position='last')
        .reset_index(drop=True)
    )

    summary_df = (
        results_df
        .groupby('k', as_index=False)
        .agg(test_accuracy_mean=('test_accuracy', 'mean'), test_accuracy_std=('test_accuracy', 'std'), test_macro_f1_mean=('test_macro_f1', 'mean'), test_macro_f1_std=('test_macro_f1', 'std'), test_auc_mean=('test_auc', 'mean'), test_auc_std=('test_auc', 'std'))
    )

    if logger is not None and output_dir is not None:
        results_df.to_csv(output_dir / 'results.csv', index=False)
        aggregate_df.to_csv(output_dir / 'summary.csv', index=False)
        logger.info(f"Saved results to {output_dir / 'results.csv'}")
        logger.info(f"Saved summary to {output_dir / 'summary.csv'}")
        logger.info('\n' + aggregate_df.to_string(index=False))

    return results_df, aggregate_df