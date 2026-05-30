import sys
import contextlib
import io
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download

'''
We chose UniMed-CLIP as one of our two generalist medical VLMs because it was trained on over 5.3 million image-text pairs across six diverse imaging modalities (including ultrasound imaging).
See: https://github.com/mbzuai-oryx/UniMed-CLIP
'''

# Loads UniMed-CLIP from the vendored source tree instead of the installed open_clip package.
def _load_unimed_open_clip(project_root):
    unimed_src = Path(project_root) / 'external' / 'UniMed-CLIP' / 'src'

    if not unimed_src.exists():
        raise FileNotFoundError(
            f'unimed source not found: {unimed_src}\n'
            'run: git clone https://github.com/mbzuai-oryx/UniMed-CLIP.git external/UniMed-CLIP'
        )

    if str(unimed_src) not in sys.path:
        sys.path.insert(0, str(unimed_src))

    for module_name in list(sys.modules.keys()):
        if module_name == 'open_clip' or module_name.startswith('open_clip.'):
            del sys.modules[module_name]

    import open_clip as unimed_open_clip

    return unimed_open_clip

def _resolve_unimed_project_root(project_root=None):
    repo_root = Path(__file__).resolve().parents[5]

    if project_root is not None:
        candidate = Path(project_root)

        if (candidate / 'external' / 'UniMed-CLIP' / 'src').exists():
            return candidate

    if (repo_root / 'external' / 'UniMed-CLIP' / 'src').exists():
        return repo_root

    cwd = Path.cwd()

    if (cwd / 'external' / 'UniMed-CLIP' / 'src').exists():
        return cwd

    if (cwd.parent / 'external' / 'UniMed-CLIP' / 'src').exists():
        return cwd.parent

    # Error.
    raise FileNotFoundError(
        f'unimed source not found in any candidate location:\n'
        f'- {repo_root / "external" / "UniMed-CLIP" / "src"}\n'
        f'- {cwd / "external" / "UniMed-CLIP" / "src"}\n'
        f'- {cwd.parent / "external" / "UniMed-CLIP" / "src"}\n'
        'run: git clone https://github.com/mbzuai-oryx/UniMed-CLIP.git external/UniMed-CLIP'
    )

def _resolve_unimed_checkpoint(project_root):
    checkpoint_dir = project_root / 'models' / 'unimed'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / 'unimed-clip-vit-b16.pt'

    if checkpoint_path.exists():
        return checkpoint_path

    return Path(hf_hub_download(repo_id='UzairK/unimed-clip-vit-b16', filename='unimed-clip-vit-b16.pt', local_dir=checkpoint_dir, local_dir_use_symlinks=False))


# UniMed-CLIP checkpoints need weights_only = False with newer PyTorch versions.
def _create_unimed_model_with_torch_load_patch(unimed_open_clip, model_name, checkpoint_path, device, mean, std, text_encoder_name):
    original_torch_load = torch.load

    def torch_load_compat(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False

        return original_torch_load(*args, **kwargs)

    try:
        torch.load = torch_load_compat

        model, _, preprocess = unimed_open_clip.create_model_and_transforms(model_name, str(checkpoint_path), precision='fp32', device=device, force_quick_gelu=True, pretrained_image=False, mean=mean, std=std, inmem=True, text_encoder_name=text_encoder_name)

    finally:
        torch.load = original_torch_load

    return model, preprocess

@contextlib.contextmanager
def _quiet_unimed_load(enabled=True):
    if not enabled:
        yield
        return

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            from transformers.utils import logging as transformers_logging
            previous_verbosity = transformers_logging.get_verbosity()
            transformers_logging.set_verbosity_error()
        except Exception:
            transformers_logging = None
            previous_verbosity = None

        try:
            yield
        finally:
            if transformers_logging is not None:
                transformers_logging.set_verbosity(previous_verbosity)

def load_unimedclip(device='cuda', project_root=None, quiet=True):
    project_root = _resolve_unimed_project_root(project_root)

    with _quiet_unimed_load(quiet):
        unimed_open_clip = _load_unimed_open_clip(project_root)
        checkpoint_path = _resolve_unimed_checkpoint(project_root)

        model_name = 'ViT-B-16-quickgelu'
        text_encoder_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)

        model, preprocess = _create_unimed_model_with_torch_load_patch(unimed_open_clip=unimed_open_clip, model_name=model_name, checkpoint_path=checkpoint_path, device=device, mean=mean, std=std, text_encoder_name=text_encoder_name)
        tokenizer = unimed_open_clip.HFTokenizer(text_encoder_name, context_length=256)

    model.eval().float()

    return model, preprocess, tokenizer

def make_unimedclip_loader(device = 'cuda', project_root = None):
    def loader():
        model, preprocess, _ = load_unimedclip(device=device, project_root=project_root)
        return model, preprocess

    return loader
