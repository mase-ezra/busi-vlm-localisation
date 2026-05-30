#### A Unified Framework for Breast Ultrasound Lesion Classification and Segmentation

In our project, we developed a reusable preprocessing pipeline with caliper and annotation removal, evaluated vision-language models (OpenAI CLIP, BiomedCLIP, and UniMed-CLIP) for BUSI breast ultrasound classification, and performed BUSSAM lesion localization. Our aim is to evaluate the suitability of generalist and medical-domain VLMs for breast ultrasound classification. This project also serves as a launchpad for future research using lightweight adaptation techniques.

###### Setup

Clone the repository -
```bash
git clone https://github.com/mase-ezra/busi-vlm-localisation.git
cd busi-vlm-localisation
python -m venv .venv
.venv\Scripts\activate
```

Install CUDA PyTorch first if using an NVIDIA GPU -
```bash
pip install -r requirements-gpu.txt
```

Install the remaining dependencies -
```bash
pip install -r requirements.txt
```

Create a `.env` file -
```env
kaggle_username=your_kaggle_username
kaggle_api_key=your_kaggle_api_key
huggingface_token=your_huggingface_token
azure_openai_endpoint=your_azure_openai_endpoint
azure_openai_api_key=your_azure_openai_api_key
```

###### Pipeline

Run all of the notebooks in order.
```text
1. notebooks/01-preprocessing.ipynb
2. notebooks/02-prompt-ensembling.ipynb
3. notebooks/03-vlm-classification.ipynb
4. notebooks/04-train-bussam.ipynb
```

###### Training Settings
```text
Few-shot classification:
- shots per class: 1, 2, 4, 8, 16, 32
- seeds: 1-10
- linear probe max_iter: 5000
- LoRA epochs: 100
- LoRA batch size: 8
- LoRA gradient accumulation: 4
- LoRA patience: 18
- LoRA head learning rate: 1e-3
- LoRA adapter learning rate: 1e-4
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.1
- LoRA layers: all vision transformer layers

BUSSAM localization:
- epochs: 20
- batch size: 8
- base learning rate: 0.0005
- SAM backbone: ViT-B
- encoder input size: 256
- low image size: 128
```

###### Future Extensions
- Evaluate on larger breast ultrasound datasets such as BUS-BRA to improve external validity.
- Fine-tune and rerank prompting strategies to improve zero-shot performance.
- Compare against CNN baselines such as ResNet, DenseNet, and EfficientNet models.
- Further optimize LoRA per model; reducing trainable parameters by injecting into fewer vision encoder layers did not improve results in our tests.

```bibtex
@software{group9_1_2026_busi_vlm_localisation,
  author = {Webb, Jet and Wright, Gulliver and Zhang, Justin and Chan, Timothy and Jeffrey, Mason},
  title = {A Unified Framework for Breast Image Ultrasound Lesion Classification and Segmentation},
  year = {2026},
  institution = {University of Technology Sydney},
  course = {49275: Neural Networks and Fuzzy Logic},
  assessment = {Research Paper},
  note = {Autumn 2026}
}
```