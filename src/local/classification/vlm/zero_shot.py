'''
Zero-shot CLIP classification helpers for all three CLIP variants using prompt ensembling.

See: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/zero_shot_classifier.py
'''

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)

# Move tokenized text to device.
def move_tokenized_to_device(tokenized, device):
    if hasattr(tokenized, 'to'):
        return tokenized.to(device)
    
    return tokenized

# Build text embeddings for each class.
def build_text_embeddings(model, tokenizer, prompt_registry, class_names, device='cuda', tokenizer_kwargs=None):
    tokenizer_kwargs = tokenizer_kwargs or {}
    zeroshot_weights = []

    for class_name in class_names:
        prompts = prompt_registry[class_name]

        tokenized = tokenizer(prompts, **tokenizer_kwargs)
        tokenized = move_tokenized_to_device(tokenized, device)

        with torch.no_grad():
            text_features = model.encode_text(tokenized)

            # Normalise each prompt embedding.
            text_features = F.normalize(text_features, dim=-1)

            # Average the eight prompts into one class prototype.
            class_embedding = text_features.mean(dim=0)

            # Normalise after averaging.
            class_embedding = F.normalize(class_embedding, dim=0)

        zeroshot_weights.append(class_embedding)

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    return zeroshot_weights

# Compute zero-shot predictions using prompt ensemble averaging.
def predict_from_image_features(image_features, text_embeddings, model):
    # Ensure image features are normalised.
    image_features = F.normalize(image_features, dim=-1)

    # Cosine similarity scores for image-to-class matching.
    similarity_scores = image_features @ text_embeddings

    # Scale similarities using CLIP's learned scale and choose the best match.
    logit_scale = model.logit_scale.exp()
    logits = logit_scale * similarity_scores

    probabilities = torch.softmax(logits, dim = 1)
    predictions = logits.argmax(dim = 1)

    return (predictions.detach().cpu().numpy(), probabilities.detach().cpu().numpy())

# End-to-end zero-shot prediction from images.
def zero_shot_predict(model, images, text_embeddings, device = 'cuda'):
    images = images.to(device)

    with torch.no_grad():
        # Encode and normalize images.
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim = -1)

        predictions, probabilities = predict_from_image_features(image_features, text_embeddings, model)

    return predictions, probabilities

# Compute comprehensive classification metrics.
def compute_classification_metrics(true_labels, predictions, probabilities, class_names):
    metrics = {}
    labels = np.arange(len(class_names))
    
    # Overall metrics.
    metrics['accuracy'] = accuracy_score(true_labels, predictions)
    metrics['balanced_accuracy'] = balanced_accuracy_score(true_labels, predictions)
    metrics['macro_f1'] = f1_score(true_labels, predictions, average='macro', zero_division=0)
    metrics['weighted_f1'] = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    
    # AUC.
    try:
        if len(class_names) == 2:
            metrics['auc'] = roc_auc_score(true_labels, probabilities[:, 1])
        else:
            metrics['auc'] = roc_auc_score(true_labels, probabilities, multi_class='ovr', average='macro')
    except ValueError:
        metrics['auc'] = np.nan
    
    # Per-class metrics.
    precision_per_class = precision_score(true_labels, predictions, labels=labels, average=None, zero_division=0)
    recall_per_class = recall_score(true_labels, predictions, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(true_labels, predictions, labels=labels, average=None, zero_division=0)
    
    for idx, class_name in enumerate(class_names):
        metrics[f'{class_name}_precision'] = precision_per_class[idx]
        metrics[f'{class_name}_recall'] = recall_per_class[idx]
        metrics[f'{class_name}_f1'] = f1_per_class[idx]
    
    return metrics

# Class for zero-shot CLIP evaluation.
class ZeroShotEvaluator:
    def __init__(self, model, preprocess, tokenizer, prompt_registry, class_names, device='cuda', tokenizer_kwargs = None):
        self.model = model.to(device)
        self.model.eval()

        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.prompt_registry = prompt_registry
        self.class_names = class_names
        self.device = device
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.text_embeddings = None
        
    # Build text embeddings from prompt registry.
    def build_text_embeddings(self, verbose=True):
        self.text_embeddings = build_text_embeddings(self.model, self.tokenizer, self.prompt_registry, self.class_names, device=self.device, tokenizer_kwargs=self.tokenizer_kwargs)

        if verbose:
            print(f'class text prototypes - {self.text_embeddings.shape}')

        return self.text_embeddings
    
    # Encode images from the dataframe.
    def encode_images(self, dataframe, batch_size=32, description='encoding images'):
        from .helpers import encode_images_batch
        
        image_features = encode_images_batch(self.model, self.preprocess, dataframe, device=self.device, batch_size=batch_size, description=description)
        return image_features
    
    # Run full evaluation pipeline: encode images, predict, compute metrics.
    def evaluate(self, dataframe, batch_size=32, description='encoding images'):
        if self.text_embeddings is None:
            self.build_text_embeddings(verbose=False)

        image_features = self.encode_images(dataframe, batch_size=batch_size, description=description)

        true_labels = dataframe['label_index'].values

        predictions, probabilities = predict_from_image_features(image_features.to(self.device), self.text_embeddings, self.model)

        base_class_names = []

        for name in self.class_names:
            if ' tumor' in name:
                base_class_names.append(name.replace(' tumor', ''))

            elif ' scan' in name:
                base_class_names.append(name.replace(' scan', ''))

            else:
                base_class_names.append(name)

        metrics = compute_classification_metrics(true_labels, predictions, probabilities, base_class_names)

        return metrics, predictions, probabilities
    
    # Show the zero-shot evaluation results.
    def print_results(self, metrics, model_name='CLIP'):
        print(f'\nZero-Shot Results: {model_name}')
        print(f"accuracy - {metrics['accuracy']:.4f}")
        print(f"balanced Accuracy - {metrics['balanced_accuracy']:.4f}")
        print(f"macro F1 - {metrics['macro_f1']:.4f}")
        print(f"AUC - {metrics['auc']:.4f}")
        print(f"\nper-class F1 -")
        
        # Extract base class names from metrics keys.
        for key in metrics.keys():
            if key.endswith('_f1') and key not in ['macro_f1', 'weighted_f1']:
                class_name = key.replace('_f1', '')
                print(f"  {class_name}: {metrics[key]:.4f}")