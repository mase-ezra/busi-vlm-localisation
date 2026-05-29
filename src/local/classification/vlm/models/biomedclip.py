import torch
import open_clip

'''
We chose Microsoft’s BiomedCLIP as a biomedical VLM because it was pretrained on PMC-15M (a dataset with over 15 million biomedical image-text pairs collected from PubMed articles).
See: https://github.com/microsoft/BiomedCLIP_data_pipeline
'''

# Loads the BiomedCLIP model.
def load_biomedclip(device="cuda"):
    model_id = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_id, device=device)
    tokenizer = open_clip.get_tokenizer(model_id)
    model = model.float()
    return model, preprocess_train, preprocess_val, tokenizer

def make_biomedclip_loader(device="cuda"):
    def loader():
        model, _, preprocess_val, _ = load_biomedclip(device=device)
        return model, preprocess_val

    return loader
