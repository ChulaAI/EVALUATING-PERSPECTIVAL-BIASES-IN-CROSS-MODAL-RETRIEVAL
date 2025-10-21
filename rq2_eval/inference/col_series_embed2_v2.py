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

from colpali_engine.models import ColQwen2, ColQwen2Processor, ColQwen2_5, ColQwen2_5_Processor

# --- Configuration ---
model_name = "vidore/colqwen2-v1.0"
model_real_name = model_name.split("/")[-1]

# Initialize your GME model
model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # or "mps" if on Apple Silicon
    ).eval()

processor = ColQwen2Processor.from_pretrained(model_name)

# Load the new dataset
dataset_name = "Chula-AI/association_bias_benchmark"
print(f"Loading dataset: {dataset_name}")
dataset = load_dataset(dataset_name, name="image_metadata", split="train")

print("Nah, I would win.")

eval_size = min(np.inf, len(dataset))
eval_dataset = dataset.select(range(eval_size))

# Define the start and end indices for processing (can be adjusted for batching/resuming)
start_index = 0
end_index = eval_size # Process all items by default

embeddings_list = [] # Store all processed embeddings

save_dir = os.path.join("rq2_eval", "embeddings")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"image_text_embeddings_{model_real_name}_{start_index}_{end_index}.pkl")

print(f"Processing entries from index {start_index} to {end_index} for dataset '{dataset_name}'...")
print(f"Total dataset size: {len(eval_dataset)}")

# Process entries within the specified range
for i in tqdm(range(start_index, min(end_index, len(eval_dataset))), desc="Processing entries"):
    current_entry = eval_dataset[i]

    # Initialize a dictionary to store embeddings for the current entry
    entry_embeddings = {
        'concept_id': current_entry.get('concept_id', None),
        'image_id': current_entry.get('image_id', None),
        'image_key': current_entry.get('image_id', None), # Keep for backward compatibility
        'index_in_dataset': i, # Keep track of the original index for reference
        'concept': current_entry.get('concept', None),
        'concept_in_native': current_entry.get('concept_in_native'),
        'concept_country': current_entry.get('concept_country', None),
        'country': current_entry.get('country', None),
        'title': current_entry.get('title', None)
    }

    try:
        with torch.no_grad():
            # 1. Get image embedding
            # The 'image' column in the dataset contains PIL Image objects
            # No, don't collect image embeddings
            # if current_entry['image'] is not None:
            #     # image_embeddings = gme.get_image_embeddings(images=[current_entry['image']])
            #     batch_images = processor.process_images([current_entry['image']]).to(model.device)
            #     image_embeddings = model(**batch_images)
            #     entry_embeddings['image_embedding'] = image_embeddings.cpu().float().numpy()
            # else:
            #     print(f"Warning: No image found for entry index {i}")
            #     entry_embeddings['image_embedding'] = None

            # 2. Get text embeddings for ALL available text fields
            text_embeddings_dict = {}
            
            # Check if 'concept' field has text content (seems to be the main text field based on your structure)
            if 'concept' in current_entry and current_entry['concept'] is not None:
                if isinstance(current_entry['concept'], str):
                    # Single string concept
                    # concept_text_embeddings = gme.get_text_embeddings(texts=[current_entry['concept']])
                    batch_queries = processor.process_queries([current_entry['concept']]).to(model.device)
                    concept_text_embeddings = model(**batch_queries)
                    concept_in_native_queries = processor.process_queries([current_entry['concept_in_native']]).to(model.device)
                    concept_in_native_embeddings = model(**concept_in_native_queries)
                    text_embeddings_dict['concept_embedding'] = concept_text_embeddings.cpu().float().numpy()
                    text_embeddings_dict['translated_concept_embedding'] = concept_in_native_embeddings.cpu().float().numpy()
                elif isinstance(current_entry['concept'], dict):
                    # Dictionary of concepts by language/country
                    # print("Raise isinstance(current_entry['concept'], dict):")
                    raise Exception("Sorry, no numbers below zero") 

            entry_embeddings['text_embeddings'] = text_embeddings_dict
            
            if not text_embeddings_dict:
                print(f"Warning: No valid text embeddings generated for entry index {i}")

        embeddings_list.append(entry_embeddings)

    except Exception as e:
        print(f"Error processing entry {i}: {str(e)}")
        # Still save the entry with available metadata, even if embedding generation failed
        entry_embeddings['image_embedding'] = None
        entry_embeddings['text_embeddings'] = {}
        entry_embeddings['error'] = str(e)
        embeddings_list.append(entry_embeddings)

    # # Clean up GPU memory periodically
    # if i % 100 == 0:
    #     torch.cuda.empty_cache()

# Final cleanup
torch.cuda.empty_cache()

# Save all collected embeddings
print(f"Saving embeddings to: {save_path}")
with open(save_path, 'wb') as f:
    pickle.dump(embeddings_list, f)

print(f"Processing complete. Embeddings saved to: {save_path}")
print(f"Total entries processed: {len(embeddings_list)}")

# Print summary statistics
successful_image_embeddings = sum(1 for entry in embeddings_list if entry.get('image_embedding') is not None)
successful_text_embeddings = sum(1 for entry in embeddings_list if entry.get('text_embeddings') and len(entry['text_embeddings']) > 0)

print(f"Successful image embeddings: {successful_image_embeddings}/{len(embeddings_list)}")
print(f"Successful text embeddings: {successful_text_embeddings}/{len(embeddings_list)}")

# Show sample of what was saved
if embeddings_list:
    print("\nSample entry structure:")
    sample_entry = embeddings_list[0]
    for key, value in sample_entry.items():
        if key == 'text_embeddings' and isinstance(value, dict):
            print(f"  {key}: {list(value.keys())}")
        elif isinstance(value, np.ndarray):
            print(f"  {key}: numpy array shape {value.shape}")
        else:
            print(f"  {key}: {type(value)}")

# print("embeddings_list[0]['image_embedding']",embeddings_list[0]['text_embedding']['concept_embedding'])