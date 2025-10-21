import torch
import faiss
import torch.nn as nn
from collections import defaultdict

from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import pickle
import os
from datetime import datetime
import pandas as pd
from colpali_engine.models import ColQwen2, ColQwen2Processor
# from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# Load model and processor
model_name = "vidore/colqwen2-v1.0"
model_name_only = model_name.split("/")[-1]
# model = ColQwen2_5.from_pretrained(
#     "Metric-AI/ColQwen2.5-7b-multilingual-v1.0",
#     torch_dtype=torch.bfloat16,
#     device_map="cuda:0",  # or "mps" if on Apple Silicon
# ).eval()
processor = ColQwen2Processor.from_pretrained(model_name)

# Load embeddings with new schema
embedding_dir_path = os.path.join("rq1_eval", "embeddings")
file_name = "image_text_embeddings_colqwen2-v1.0_0_3600.pkl"
with open(embedding_dir_path + file_name, 'rb') as f:
    embeddings = pickle.load(f)

def to_torch_tensor(data):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        return data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

# Extract embeddings from new schema
all_candidate_embedding = []  # candidates from images
all_query_embeddings = []  # queries from images (same as candidates in this case)

print("Extracting embeddings from new schema...")
for i, entry in enumerate(tqdm(embeddings, desc="Processing entries")):
    # Extract image embedding (used as query)
    if 'image_embedding' in entry:
        image_emb = to_torch_tensor(entry['image_embedding'])
        # 
        all_query_embeddings.append(image_emb)
        # # Also use image as candidate, nah we don't.
        # all_candidate_embedding.append(image_emb)
    else:
        print(f"Warning: No image_embedding found in entry {i}")
        continue
    
    # Extract text embeddings from all languages (used as candidates)
    if 'text_embeddings' in entry:
        text_embeddings = entry['text_embeddings']
        for lang_key, lang_embeddings_list in text_embeddings.items():
            # lang_embeddings_list is a list of embeddings for that language
            for caption_emb_array in lang_embeddings_list:
                caption_emb = to_torch_tensor(caption_emb_array)
                all_candidate_embedding.append(caption_emb)
    else:
        print(f"Warning: No text_embeddings found in entry {i}")

print(f"Total image embeddings (candidates): {len(all_candidate_embedding)}")
print(f"Total query embeddings: {len(all_query_embeddings)}")

TARGET_EMBEDDING_DIM = 128

processed_image_ann_vectors = []
valid_image_indices = []

print(f"Standardizing image embeddings to {TARGET_EMBEDDING_DIM} dimensions...")

for i, img_emb_raw in enumerate(tqdm(all_candidate_embedding, desc="Processing images for ANN")):
    if img_emb_raw.numel() == 0:
        print(f"Warning: Image {i} has an empty embedding. Skipping for ANN indexing.")
        continue
    
    # Attempt to remove singleton batch dimension (dim=0).
    img_emb_processed_for_mean = img_emb_raw.squeeze(0)

    if img_emb_processed_for_mean.numel() == 0:
        print(f"DEBUG SKIP: Image {i} - Tensor is empty after squeeze(0). Original shape: {img_emb_raw.shape}")
        print(f"Warning: Image {i} has an empty embedding after initial processing. Skipping.")
        continue

    # Check the dimension and last dimension of the processed tensor
    is_2d_and_correct_dim = (img_emb_processed_for_mean.dim() == 2 and img_emb_processed_for_mean.shape[1] == TARGET_EMBEDDING_DIM)
    is_1d_and_correct_dim = (img_emb_processed_for_mean.dim() == 1 and img_emb_processed_for_mean.shape[0] == TARGET_EMBEDDING_DIM)

    if not (is_2d_and_correct_dim or is_1d_and_correct_dim):
        print(f"\n--- DEBUGGING SKIP REASON for Image {i} ---")
        print(f"  Original raw shape: {img_emb_raw.shape}")
        print(f"  Shape after squeeze(0): {img_emb_processed_for_mean.shape}")
        print(f"  Dimensions: {img_emb_processed_for_mean.dim()}")
        if img_emb_processed_for_mean.dim() > 0:
            print(f"  Size of last dimension: {img_emb_processed_for_mean.shape[-1]}")
        else:
            print(f"  Tensor has no dimensions (scalar or 0D).")
        print(f"  Expected TARGET_EMBEDDING_DIM: {TARGET_EMBEDDING_DIM}")
        print(f"  Condition 'is_2d_and_correct_dim': {is_2d_and_correct_dim}")
        print(f"  Condition 'is_1d_and_correct_dim': {is_1d_and_correct_dim}")
        print(f"--- END DEBUGGING SKIP REASON ---")
        print(f"Warning: Image {i} has truly unexpected embedding shape after initial processing: {img_emb_processed_for_mean.shape}. Skipping.")
        continue

    # If we reach here, the shape is either [N, D] or [D].
    if img_emb_processed_for_mean.dim() == 2:
        single_vector_for_ann = torch.mean(img_emb_processed_for_mean, dim=0).to(torch.float32)
    else:  # It must be 1D, [D]
        single_vector_for_ann = img_emb_processed_for_mean.to(torch.float32)

    processed_image_ann_vectors.append(single_vector_for_ann)
    valid_image_indices.append(i)

if not processed_image_ann_vectors:
    raise ValueError("No valid image ANN vectors could be generated. Check your all_candidate_embedding data and filter.")

image_ann_vectors = torch.stack(processed_image_ann_vectors).cpu().numpy()

# Store the original multi-vectors for reranking
image_embedding_map = {}
for original_idx in valid_image_indices:
    original_full_multi_vector_raw = all_candidate_embedding[original_idx]
    image_embedding_map[original_idx] = original_full_multi_vector_raw

# Create FAISS index
d = image_ann_vectors.shape[1]
index = faiss.IndexFlatL2(d)
index.add(image_ann_vectors)
print(f"FAISS index created with {index.ntotal} vectors.")

# Process Query Embeddings
processed_query_ann_vectors = []
for q_emb_raw in tqdm(all_query_embeddings, desc="Processing queries for ANN"):
    if q_emb_raw.numel() == 0:
        print(f"Warning: Query has empty embedding. Using zero vector for ANN.")
        processed_query_ann_vectors.append(torch.zeros(TARGET_EMBEDDING_DIM, dtype=torch.float32))
        continue

    q_emb_processed_for_mean = q_emb_raw.squeeze(0)

    is_2d_and_correct_dim_q = (q_emb_processed_for_mean.dim() == 2 and q_emb_processed_for_mean.shape[1] == TARGET_EMBEDDING_DIM)
    is_1d_and_correct_dim_q = (q_emb_processed_for_mean.dim() == 1 and q_emb_processed_for_mean.shape[0] == TARGET_EMBEDDING_DIM)

    if not (is_2d_and_correct_dim_q or is_1d_and_correct_dim_q):
        print(f"\n--- DEBUGGING SKIP REASON for Query ---")
        print(f"  Original raw shape: {q_emb_raw.shape}")
        print(f"  Shape after squeeze(0): {q_emb_processed_for_mean.shape}")
        print(f"  Dimensions: {q_emb_processed_for_mean.dim()}")
        if q_emb_processed_for_mean.dim() > 0:
            print(f"  Size of last dimension: {q_emb_processed_for_mean.shape[-1]}")
        else:
            print(f"  Tensor has no dimensions (scalar or 0D).")
        print(f"  Expected TARGET_EMBEDDING_DIM: {TARGET_EMBEDDING_DIM}")
        print(f"  Condition 'is_2d_and_correct_dim_q': {is_2d_and_correct_dim_q}")
        print(f"  Condition 'is_1d_and_correct_dim_q': {is_1d_and_correct_dim_q}")
        print(f"--- END DEBUGGING SKIP REASON ---")
        print(f"Warning: Query has truly unexpected embedding shape after initial processing: {q_emb_processed_for_mean.shape}. Using zero vector for ANN.")
        processed_query_ann_vectors.append(torch.zeros(TARGET_EMBEDDING_DIM, dtype=torch.float32))
        continue

    if q_emb_processed_for_mean.dim() == 2:
        single_vector_for_ann = torch.mean(q_emb_processed_for_mean, dim=0).to(torch.float32)
    else:  # It must be 1D, [D]
        single_vector_for_ann = q_emb_processed_for_mean.to(torch.float32)

    processed_query_ann_vectors.append(single_vector_for_ann)

if not processed_query_ann_vectors:
    raise ValueError("No valid query ANN vectors could be generated.")

query_ann_vectors = torch.stack(processed_query_ann_vectors).cpu().numpy()

# Retrieval and Reranking
K_ANN_CANDIDATES = 100
K_FINAL_RETRIEVAL = 100

# Create mapping to track original entry indices
# Since we now have multiple candidates per entry (image + multiple captions),
# we need to map back to the original entry
candidate_to_entry_map = {}
candidate_idx = 0

for entry_idx, entry in enumerate(embeddings):
    # Image embedding
    if 'image_embedding' in entry:
        candidate_to_entry_map[candidate_idx] = {
            'entry_idx': entry_idx,
            'type': 'image',
            'image_key': entry.get('image_key', f'entry_{entry_idx}')
        }
        candidate_idx += 1
    
    # Text embeddings
    if 'text_embeddings' in entry:
        for lang_key, lang_embeddings_list in entry['text_embeddings'].items():
            for caption_idx, _ in enumerate(lang_embeddings_list):
                candidate_to_entry_map[candidate_idx] = {
                    'entry_idx': entry_idx,
                    'type': 'caption',
                    'language': lang_key,
                    'caption_idx': caption_idx,
                    'image_key': entry.get('image_key', f'entry_{entry_idx}')
                }
                candidate_idx += 1

all_search_results = []
sample_size = len(all_query_embeddings)
print(f"Total queries to process: {sample_size}")

for i in tqdm(range(len(all_query_embeddings)), desc="Retrieving and Reranking"):
    query_original_emb = all_query_embeddings[i]
    query_ann_vec = query_ann_vectors[i:i+1]

    # FAISS search
    D, I = index.search(query_ann_vec, K_ANN_CANDIDATES)
    faiss_candidate_indices = I[0]

    # Reranking
    reranked_scores = []
    for faiss_idx in faiss_candidate_indices:
        if faiss_idx == -1:
            continue
        
        original_img_idx = valid_image_indices[faiss_idx]
        image_original_emb = image_embedding_map[original_img_idx]

        # Calculate similarity score
        score = processor.score_multi_vector(query_original_emb, image_original_emb)
        
        # Get metadata about this candidate
        candidate_info = candidate_to_entry_map.get(original_img_idx, {})
        
        reranked_scores.append((score.item(), original_img_idx, candidate_info))

    # Sort by score and take top K
    reranked_scores.sort(key=lambda x: x[0], reverse=True)
    
    for rank, (score, result_id, candidate_info) in enumerate(reranked_scores[:K_FINAL_RETRIEVAL]):
        all_search_results.append({
            'query_id': i,
            'result_rank': rank + 1,
            'result_id': result_id,
            'entry_idx': candidate_info.get('entry_idx', -1),
            'candidate_type': candidate_info.get('type', 'unknown'),
            'language': candidate_info.get('language', 'N/A'),
            'image_key': candidate_info.get('image_key', 'unknown'),
            'score': float(f"{score:.4f}")
        })

# Save results
df = pd.DataFrame(all_search_results)
output_filename = f'{file_name}_results.csv'
df.to_csv(output_filename, index=False)
print(f"Results saved to: {output_filename}")
print(f"DataFrame shape: {df.shape}")
print(f"Candidate types distribution:")
print(df['candidate_type'].value_counts())
if 'language' in df.columns:
    print(f"Language distribution:")
    print(df['language'].value_counts())