import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import pickle
import os
import pandas as pd

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
# from colpali_engine.models import ColQwen2, ColQwen2Processor

# --- Configuration ---
model_name = "Metric-AI/ColQwen2.5-7b-multilingual-v1.0"

# model_name = "vidore/colqwen2.5-v0.2"
# model_name = "vidore/colqwen2-v1.0-merged"
model_name_part = model_name.split("/")[-1]

# Initialize ColQwen model
model = ColQwen2_5.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
# model = torch.compile(model)

processor = ColQwen2_5_Processor.from_pretrained(model_name)

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
save_path = os.path.join(save_dir, f"image_text_embeddings_{model_name_part}_{start_index}_{end_index}.pkl")

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
        batch_images = processor.process_images([current_entry['image']]).to(model.device)
        image_embeddings = model(**batch_images)
        entry_embeddings['image_embedding'] = image_embeddings.float().cpu().numpy()

        # 2. Get text embeddings for ALL languages
        text_embeddings_by_lang = {}
        # Iterate through the 'captions' dictionary
        # The structure is {lang_code: list_of_captions, ...}
        if 'captions' in current_entry and isinstance(current_entry['captions'], dict):
            for lang_code, captions_list in current_entry['captions'].items():
                if isinstance(captions_list, list) and captions_list:
                    # Process all captions for this language
                    lang_embeddings = []
                    for caption in captions_list:
                        batch_caption = processor.process_queries([caption]).to(model.device)
                        caption_embedding = model(**batch_caption)
                        lang_embeddings.append(caption_embedding.float().cpu().numpy())
                    
                    # Store as list to handle varying embedding shapes
                    text_embeddings_by_lang[f'caption_embedding_{lang_code}'] = lang_embeddings
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

# Optional: Load and verify a sample
# with open(save_path, 'rb') as f:
#     loaded_embeddings = pickle.load(f)
# print("\nSample loaded embedding entry:")
# print(loaded_embeddings[0])
# print("Image embedding shape:", loaded_embeddings[0]['image_embedding'].shape)
# for lang_key, embed in loaded_embeddings[0]['text_embeddings'].items():
#     print(f"{lang_key} shape: {embed.shape}")