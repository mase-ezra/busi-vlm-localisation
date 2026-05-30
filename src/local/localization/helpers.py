from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request

'''
BUSSAM is used here for BUSI segmentation/localisation. The segmentation target is background and lesion. 
Referred to the original BUSSAM paper and code so the training settings match the benchmark setup.
See: https://arxiv.org/html/2404.14837v1
See: https://github.com/bscs12/BUSSAM
'''

# Finds the main project folder so paths work from helper .py files not notebooks.
def find_project_root():
    return next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())

# A helper function to create BUSI filenames for the required BUSSAM dataset.
def clean_busi_name(name):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('_')

# Copy every BUSI image or mask into the BUSSAM dataset folders.
def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        destination.unlink()
    
    shutil.copyfile(source, destination)

# A helper function to keep BUSSAM file paths pointing to the correct folder.
def resolve_bussam_path(bussam_root, output_path):
    output_directory = Path(output_path)

    if output_directory.is_absolute():
        return output_directory
    
    return bussam_root / output_directory

# Get the path to the cloned BUSSAM repo inside the external folder.
def find_bussam_repo():
    repo_path = find_project_root()/'external'/'BUSSAM'
    return repo_path

# Used to clone the BUSSAM repository for training. You can also run the git clone manually.
def download_bussam_code():
    bussam = find_bussam_repo()
    print(f'clone: exists = {bussam.exists()} path = {bussam}')

    if bussam.exists():
        return bussam
    
    bussam.parent.mkdir(parents = True, exist_ok = True)

    # Clone the official BUSSAM repository into the local external/BUSSAM folder.
    subprocess.run(['git', 'clone', 'https://github.com/bscs12/BUSSAM.git', str(bussam)], check = True)

    return bussam

# Download the pretrained SAM ViT-B weights used to initialise BUSSAM before BUSI segmentation training.
def download_sam_vit_b_checkpoint():
    bussam = find_bussam_repo()
    checkpoint = bussam / 'checkpoints' / 'sam_vit_b_01ec64.pth'

    print(f'sam checkpoint: exists = {checkpoint.exists()} path = {checkpoint}')

    if checkpoint.exists():
        return checkpoint
    
    checkpoint.parent.mkdir(parents = True, exist_ok = True)

    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', checkpoint)

    return checkpoint

# Copies the BUSI split images and masks into the folder structure BUSSAM expects.
def prepare_busi_for_bussam():
    classes = ['benign', 'malignant']

    root = find_project_root()
    bussam = find_bussam_repo()

    split_root = root/'dataset'/'split'
    out_image = bussam/'datasets'/'BUSI'/'img'
    out_label = bussam/'datasets'/'BUSI'/'label'
    out_main = bussam/'datasets'/'MainPatient'

    # Create the dataset folders.
    out_image.mkdir(parents = True, exist_ok = True)
    out_label.mkdir(parents = True, exist_ok = True)
    out_main.mkdir(parents = True, exist_ok = True)

    print(f'dataset source = {split_root}')
    print(f'dataset classes = {classes}')

    split_lists = {'train': [], 'val': [], 'test': []}

    file_plan = []

    # Collect each image-mask pair from our existing split.
    for split in split_lists:
        for class_name in classes:
            class_directory = split_root/split/class_name

            if not class_directory.exists():
                continue

            for image_path in sorted(class_directory.glob('*.png')):
                if '_mask' in image_path.stem:
                    continue

                mask_path = image_path.with_name(f'{image_path.stem}_mask{image_path.suffix}')
                out_stem = clean_busi_name(f'{class_name}__{image_path.stem}')
                out_file = f'{out_stem}.png'

                file_plan.append({'image_src': str(image_path.relative_to(root)).replace('\\', '/'), 'mask_src': str(mask_path.relative_to(root)).replace('\\', '/'), 'output_name': out_file})

                # BUSSAM uses a different split-file prefix for test samples.
                if split == 'test':
                    split_lists[split].append(f'BUSI/{out_stem}')

                else:
                    split_lists[split].append(f'1/BUSI/{out_stem}')

    # Store the BUSSAM split-file path for each dataset split.
    split_file_paths = {}

    for split in split_lists:
        split_file_path = out_main / f'BUSI_{split}.txt'
        split_file_paths[split] = split_file_path

    for png_file in out_image.glob('*.png'):
        png_file.unlink()

    for png_file in out_label.glob('*.png'):
        png_file.unlink()

    for item in file_plan:
        image_source = root/item['image_src']
        mask_source = root/item['mask_src']
        copy_file(image_source, out_image/item['output_name'])
        copy_file(mask_source, out_label/item['output_name'])

    # Save the sample names.
    for split in split_lists:
        names = split_lists[split]
        split_file_path = split_file_paths[split]

        file_text = '\n'.join(names)

        if names:
            file_text = file_text + '\n'

        split_file_path.write_text(file_text)

    (out_main / 'class.json').write_text(json.dumps({'BUSI': 2}) + '\n')

    print(f'train/val/test = {len(split_lists["train"])}/{len(split_lists["val"])}/{len(split_lists["test"])}')

    print('files copied for BUSSAM.')

    return {
        'img_dir': out_image,
        'label_dir': out_label,
        'split_files': split_file_paths,
        'counts': {split: len(names) for split, names in split_lists.items()},
    }

# BUSSAM reads epochs and output paths from config.py. 
def set_bussam_training_settings(epochs=100, output_dir='outputs/'):
    bussam = find_bussam_repo()
    config = bussam/'utils'/'config.py'

    print(f'config: epochs={epochs} output={output_dir}')

    text = config.read_text()

    text = text.replace('pre_trained = True', 'pre_trained = False')
    text = re.sub(r'epochs\s*=\s*\d+', f'epochs = {epochs}', text)
    text = re.sub(r'output_path\s*=\s*[\'"].*?[\'"]', f'output_path = \'{output_dir}\'', text)

    config.write_text(text)

    return config

# We had to fix two import-path naming errors in the released BUSSAM code so the model could load.
def fix_bussam_import_paths():
    bussam = find_bussam_repo()

    model_dict = bussam/'models'/'model_dict.py'
    text = model_dict.read_text()

    text = text.replace('from models.segment_anything_bussam.build_sam_us import bussam_model_registry', 'from models.segment_anything_samus.build_sam_us import bussam_model_registry')

    model_dict.write_text(text)

    modeling_init = bussam/'models'/'segment_anything_samus'/'modeling'/'__init__.py'
    text = modeling_init.read_text()

    text = text.replace('from .bussam import Bussam', 'from .samus import Bussam')
    
    modeling_init.write_text(text)

    train_script = bussam/'train.py'
    text = train_script.read_text()
    text = text.replace('            print(train_loss)\n', '')
    train_script.write_text(text)

    print('BUSSAM repo imports patched')

    return {
        'model_dict': model_dict,
        'modeling_init': modeling_init,
    }

# Run a subprocess while streaming output back to notebooks immediately.
def run_streamed_subprocess(command, cwd):
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    output_lines = []

    with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env) as process:
        stdout = process.stdout

        if stdout is None:
            raise RuntimeError('BUSSAM subprocess stdout was not available.')

        for line in stdout:
            print(line, end='', flush=True)
            output_lines.append(line.rstrip('\n'))

        return_code = process.wait()

    output_text = '\n'.join(output_lines)
    result = subprocess.CompletedProcess(command, return_code, stdout=output_text)

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command, output=output_text)

    return result

# Extract per-epoch training metrics from BUSSAM's stdout.
def parse_bussam_training_history(output_text):
    history_by_epoch = {}
    train_pattern = re.compile(r'epoch:(\d+)/(\d+), train_loss:([0-9.eE+-]+)')
    val_pattern = re.compile(r'epoch:(\d+)/(\d+), val_loss:([0-9.eE+-]+), val_dice:([0-9.eE+-]+)')

    for line in output_text.splitlines():
        train_match = train_pattern.search(line)

        if train_match:
            epoch = int(train_match.group(1))
            history_by_epoch.setdefault(epoch, {'epoch': epoch, 'val_loss': None, 'val_dice': None})
            history_by_epoch[epoch]['train_loss'] = float(train_match.group(3))
            continue

        val_match = val_pattern.search(line)

        if val_match:
            epoch = int(val_match.group(1))
            history_by_epoch.setdefault(epoch, {'epoch': epoch, 'train_loss': None})
            history_by_epoch[epoch]['val_loss'] = float(val_match.group(3))
            history_by_epoch[epoch]['val_dice'] = float(val_match.group(4))

    return [history_by_epoch[epoch] for epoch in sorted(history_by_epoch)]

# Function used to train BUSSAM on the BUSI segmentation task.
# We use the same settings as the arXiv paper benchmark including 256x256 input size, 128x128 low mask, batch size 8, 100 epochs, AdamW, 5e-4 learning rate, warmup, and decay.
def train_bussam_on_busi(batch_size = 8, base_lr = 0.0005):
    bussam = find_bussam_repo()

    fix_bussam_import_paths()
    set_bussam_script_gpu('train.py', '0')

    command = [
        sys.executable, '-u', 'train.py',
        '--task', 'BUSI',
        '--modelname', 'BUSSAM',
        '--encoder_input_size', '256',
        '--low_image_size', '128',
        '--vit_name', 'vit_b',
        '--sam_ckpt', 'checkpoints/sam_vit_b_01ec64.pth',
        '--batch_size', str(batch_size),
        '--base_lr', str(base_lr),
        '--n_gpu', '1',
    ]

    result = run_streamed_subprocess(command, cwd = bussam)
    best_checkpoint = find_best_bussam_checkpoint()
    history = parse_bussam_training_history(result.stdout)

    return {'result': result, 'best_checkpoint': best_checkpoint, 'history': history}

# We select the checkpoint with the highest validation Dice score.
def find_best_bussam_checkpoint():
    bussam = find_bussam_repo()
    checkpoints = list((bussam / 'outputs').glob('**/checkpoints/BUSSAM_*.pth'))

    scored = []

    for checkpoint in checkpoints:
        match = re.search(r'BUSSAM_\d+_(\d+)_(\d+(?:\.\d+)?)\.pth$', checkpoint.name)

        if match:
            scored.append((float(match.group(2)), int(match.group(1)), checkpoint))

    if scored:
        scored.sort(reverse=True)
        return scored[0][2]

    return find_latest_bussam_checkpoint()

# Find the most recently saved BUSSAM checkpoint.
def find_latest_bussam_checkpoint():
    bussam = find_bussam_repo()
    checkpoint_files = list((bussam / 'outputs').glob('**/checkpoints/BUSSAM_*.pth'))

    if not checkpoint_files:
        return None

    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)

    return latest_checkpoint

# Change the GPU selected inside a BUSSAM script before it is run.
def set_bussam_script_gpu(script_name, gpu_id="0"):
    script = find_bussam_repo() / script_name
    text = script.read_text()

    text = re.sub(r"os\.environ\[['\"]CUDA_VISIBLE_DEVICES['\"]\]\s*=\s*['\"].*?['\"]", f"os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'", text)

    script.write_text(text)
    return script

# I set the GPU used by BUSSAM's test script.
def set_bussam_test_gpu(gpu_id="0"):
    return set_bussam_script_gpu('test.py', gpu_id)

# Update BUSSAM's test settings to use our selected checkpoint and device.
def set_bussam_test_settings(checkpoint, device="cuda", visual=True):
    bussam = find_bussam_repo()
    config = bussam / "utils" / "config.py"

    text = config.read_text()
    checkpoint = str(checkpoint).replace("\\", "/")

    # Point BUSSAM to the trained model selected for evaluation.
    text = re.sub(r"load_path\s*=\s*['\"].*?['\"]", f"load_path = '{checkpoint}'", text)
    # Set whether testing runs on the GPU or CPU.
    text = re.sub(r"device\s*=\s*['\"].*?['\"]", f"device = '{device}'", text)
    # Turn predicted-mask visualisations on or off.
    text = re.sub(r"visual\s*=\s*(True|False)", f"visual = {visual}", text)

    config.write_text(text)

    return config

# Run BUSSAM on the BUSI test split using the selected trained checkpoint.
def test_bussam_on_busi(checkpoint, batch_size = 8, device = 'cuda'):
    bussam = find_bussam_repo()
    fix_bussam_import_paths()

    # I set this to use my first GPU.
    set_bussam_test_gpu('0')

    checkpoint = Path(checkpoint)

    set_bussam_test_settings(checkpoint, device = device, visual = True)

    # Adjusted SAM ViT-B settings.
    command = [
        sys.executable, "-u", "test.py",
        "--task", "BUSI",
        "--modelname", "BUSSAM",
        "--encoder_input_size", "256",
        "--low_image_size", "128",
        "--vit_name", "vit_b",
        "--sam_ckpt", "checkpoints/sam_vit_b_01ec64.pth",
        "--batch_size", str(batch_size),
        "--n_gpu", "1",
    ]

    return run_streamed_subprocess(command, cwd = bussam)