import torch
from PIL import Image
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import pickle
import os
import pandas as pd

# Assuming gme_inference.py and GmeQwen2VL are correctly set up and in your path
# from gme_inference import GmeQwen2VL
from transformers import AutoModel
import open_clip
from multilingual_clip import pt_multilingual_clip
import transformers
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Configuration ---
model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
model_real_name = model_name.split("/")[-1]

# Load Model & Tokenizer
model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name).cuda()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# vision_model, _, vision_preprocess = open_clip.create_model_and_transforms('ViT-L/14', pretrained="laion400m_e32")
vision_model, vision_preprocess = clip.load("ViT-L/14", device=device)
# vision_model.to(device)

# Load the new dataset
dataset_name = "Tierone2025part2/crossmodal_3600"
print(f"Loading dataset: {dataset_name}")
dataset = load_dataset(dataset_name, split="train")

eval_size = min(np.inf, len(dataset))
eval_dataset = dataset.select(range(eval_size))

# Define the start and end indices for processing (can be adjusted for batching/resuming)
start_index = 0
end_index = eval_size # Process all items by default

embeddings_list = [] # Renamed to avoid conflict with `embeddings` dict later

save_dir = os.path.join("rq1_eval", "embeddings")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"image_text_embeddings_{model_real_name}_{start_index}_{end_index}.pkl")

print(f"Processing entries from index {start_index} to {end_index} for dataset '{dataset_name}'...")

# Process entries within the specified range
for i in tqdm(range(start_index, min(end_index, len(eval_dataset))), desc="Processing entries"):
    current_entry = eval_dataset[i]

    # Initialize a dictionary to store embeddings for the current entry
    entry_embeddings = {
        'image_key': current_entry['image_key'], # Keep track of the image key
        'index_in_dataset': i # Keep track of the original index for reference
    }

    with torch.no_grad():
        # 1. Get image embedding
        # The 'image' column in the new dataset directly contains PIL Image objects
        # image_embeddings = gme.get_image_embeddings(images=[current_entry['image']])
        image = vision_preprocess(current_entry['image']).unsqueeze(0).to(device)
        image_embeddings = vision_model.encode_image(image)
        entry_embeddings['image_embedding'] = image_embeddings.cpu().numpy()

        # 2. Get text embeddings for ALL languages
        text_embeddings_by_lang = {}
        # Iterate through the 'captions' dictionary
        # The structure is {lang_code: list_of_captions, ...}
        if 'captions' in current_entry and isinstance(current_entry['captions'], dict):
            for lang_code, captions_list in current_entry['captions'].items():
                if isinstance(captions_list, list) and captions_list:
                    # GME expects a list of texts, so pass the list directly
                    # lang_text_embeddings = gme.get_text_embeddings(texts=captions_list)
                    lang_text_embeddings = model.forward(captions_list, tokenizer)
                    # print("lang_text_embeddings shape: ", lang_text_embeddings.shape)
                    text_embeddings_by_lang[f'caption_embedding_{lang_code}'] = lang_text_embeddings.cpu().numpy()
                else:
                    print(f"Warning: No valid captions found for language '{lang_code}' in entry index {i}")
            entry_embeddings['text_embeddings'] = text_embeddings_by_lang
        else:
            print(f"Warning: 'captions' field not found or not a dictionary for entry index {i}. No text embeddings will be generated for this entry.")

        embeddings_list.append(entry_embeddings)

    # Clean up GPU memory
    torch.cuda.empty_cache()

# Save all collected embeddings
with open(save_path, 'wb') as f:
    pickle.dump(embeddings_list, f)

print(f"Processing complete. Embeddings saved to: {save_path}")
print(f"Total entries processed: {len(embeddings_list)}")