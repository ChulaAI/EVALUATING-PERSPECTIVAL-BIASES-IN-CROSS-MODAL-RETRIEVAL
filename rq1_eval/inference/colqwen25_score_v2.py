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
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# Load model and processor
model_name = "Metric-AI/ColQwen2.5-3b-multilingual-v1.0"
model_name_only = model_name.split("/")[-1]
# model = ColQwen2_5.from_pretrained(
#     "Metric-AI/ColQwen2.5-7b-multilingual-v1.0",
#     torch_dtype=torch.bfloat16,
#     device_map="cuda:0",  # or "mps" if on Apple Silicon
# ).eval()
processor = ColQwen2_5_Processor.from_pretrained(model_name)

# Load embeddings
embedding_dir_path = os.path.join("rq1_eval", "embeddings")
file_name = "image_text_embeddings_ColQwen2.5-3b-multilingual-v1.0_0_3600.pkl"
with open(os.path.join(embedding_dir_path, file_name), 'rb') as f:
    embeddings = pickle.load(f)

def to_torch_tensor(data):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        return data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

TARGET_EMBEDDING_DIM = 128

# Extract all caption embeddings from all languages as candidates
all_caption_embeddings = []
caption_metadata = []  # Store metadata for each caption

print("Extracting caption embeddings from all languages...")
for entry_idx, entry in enumerate(tqdm(embeddings, desc="Processing entries")):
    if 'text_embeddings' in entry:
        for lang_key, lang_embeddings in entry['text_embeddings'].items():
            # Extract language code from key (e.g., 'caption_embedding_en' -> 'en')
            lang_code = lang_key.replace('caption_embedding_', '')
            
            for caption_idx, caption_emb in enumerate(lang_embeddings):
                all_caption_embeddings.append(to_torch_tensor(caption_emb))
                caption_metadata.append({
                    'entry_idx': entry_idx,
                    'language': lang_code,
                    'caption_idx': caption_idx,
                    'image_key': entry.get('image_key', f'entry_{entry_idx}')
                })

# Extract image embeddings for queries
all_query_embeddings = []
query_metadata = []

print("Extracting image embeddings for queries...")
for entry_idx, entry in enumerate(tqdm(embeddings, desc="Processing queries")):
    if 'image_embedding' in entry:
        all_query_embeddings.append(to_torch_tensor(entry['image_embedding']))
        query_metadata.append({
            'entry_idx': entry_idx,
            'image_key': entry.get('image_key', f'entry_{entry_idx}')
        })

# Process caption embeddings for ANN
processed_caption_ann_vectors = []
valid_caption_indices = []

print(f"Standardizing caption embeddings to {TARGET_EMBEDDING_DIM} dimensions...")

for i, cap_emb_raw in enumerate(tqdm(all_caption_embeddings, desc="Processing captions for ANN")):
    if cap_emb_raw.numel() == 0:
        print(f"Warning: Caption {i} has an empty embedding. Skipping for ANN indexing.")
        continue
    
    # Remove singleton batch dimension
    cap_emb_processed = cap_emb_raw.squeeze(0)
    
    if cap_emb_processed.numel() == 0:
        print(f"Warning: Caption {i} has an empty embedding after processing. Skipping.")
        continue
    
    # Check dimensions
    is_2d_and_correct_dim = (cap_emb_processed.dim() == 2 and cap_emb_processed.shape[1] == TARGET_EMBEDDING_DIM)
    is_1d_and_correct_dim = (cap_emb_processed.dim() == 1 and cap_emb_processed.shape[0] == TARGET_EMBEDDING_DIM)
    
    if not (is_2d_and_correct_dim or is_1d_and_correct_dim):
        print(f"Warning: Caption {i} has unexpected embedding shape: {cap_emb_processed.shape}. Skipping.")
        continue
    
    # Convert to single vector
    if cap_emb_processed.dim() == 2:
        single_vector_for_ann = torch.mean(cap_emb_processed, dim=0).to(torch.float32)
    else:
        single_vector_for_ann = cap_emb_processed.to(torch.float32)
    
    processed_caption_ann_vectors.append(single_vector_for_ann)
    valid_caption_indices.append(i)

if not processed_caption_ann_vectors:
    raise ValueError("No valid caption ANN vectors could be generated.")

caption_ann_vectors = torch.stack(processed_caption_ann_vectors).cpu().numpy()

# Store original caption embeddings for reranking
caption_embedding_map = {}
for original_idx in valid_caption_indices:
    original_caption_emb = all_caption_embeddings[original_idx]
    caption_embedding_map[original_idx] = original_caption_emb

# Create FAISS index
d = caption_ann_vectors.shape[1]
index = faiss.IndexFlatL2(d)
index.add(caption_ann_vectors)
print(f"FAISS index created with {index.ntotal} vectors.")

# Process query embeddings
processed_query_ann_vectors = []
for q_emb_raw in tqdm(all_query_embeddings, desc="Processing queries for ANN"):
    if q_emb_raw.numel() == 0:
        print(f"Warning: Query has empty embedding. Using zero vector for ANN.")
        processed_query_ann_vectors.append(torch.zeros(TARGET_EMBEDDING_DIM, dtype=torch.float32))
        continue
    
    q_emb_processed = q_emb_raw.squeeze(0)
    
    is_2d_and_correct_dim_q = (q_emb_processed.dim() == 2 and q_emb_processed.shape[1] == TARGET_EMBEDDING_DIM)
    is_1d_and_correct_dim_q = (q_emb_processed.dim() == 1 and q_emb_processed.shape[0] == TARGET_EMBEDDING_DIM)
    
    if not (is_2d_and_correct_dim_q or is_1d_and_correct_dim_q):
        print(f"Warning: Query has unexpected embedding shape: {q_emb_processed.shape}. Using zero vector for ANN.")
        processed_query_ann_vectors.append(torch.zeros(TARGET_EMBEDDING_DIM, dtype=torch.float32))
        continue
    
    if q_emb_processed.dim() == 2:
        single_vector_for_ann = torch.mean(q_emb_processed, dim=0).to(torch.float32)
    else:
        single_vector_for_ann = q_emb_processed.to(torch.float32)
    
    processed_query_ann_vectors.append(single_vector_for_ann)

if not processed_query_ann_vectors:
    raise ValueError("No valid query ANN vectors could be generated.")

query_ann_vectors = torch.stack(processed_query_ann_vectors).cpu().numpy()

# Retrieval and Reranking
K_ANN_CANDIDATES = 100
K_FINAL_RETRIEVAL = 100

all_search_results = []
sample_size = len(all_query_embeddings)
print(f"Total queries to process: {sample_size}")

for i in tqdm(range(len(all_query_embeddings)), desc="Retrieving and Reranking"):
    query_original_emb = all_query_embeddings[i]
    query_ann_vec = query_ann_vectors[i:i+1]
    
    # Search for candidates
    D, I = index.search(query_ann_vec, K_ANN_CANDIDATES)
    faiss_candidate_indices = I[0]
    
    reranked_scores = []
    for faiss_idx in faiss_candidate_indices:
        if faiss_idx == -1:
            continue
        
        original_caption_idx = valid_caption_indices[faiss_idx]
        caption_original_emb = caption_embedding_map[original_caption_idx]
        
        # Get metadata for this caption
        caption_meta = caption_metadata[original_caption_idx]
        
        # Calculate score
        score = processor.score_multi_vector(query_original_emb, caption_original_emb)
        reranked_scores.append((score.item(), original_caption_idx, caption_meta))
    
    # Sort by score (descending)
    reranked_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Get query metadata
    query_meta = query_metadata[i]
    
    # Store results
    for rank, (score, caption_idx, caption_meta) in enumerate(reranked_scores[:K_FINAL_RETRIEVAL]):
        all_search_results.append({
            'query_id': i,
            'query_image_key': query_meta['image_key'],
            'result_rank': rank + 1,
            'result_id': caption_meta['entry_idx'],
            'result_image_key': caption_meta['image_key'],
            'language': caption_meta['language'],
            'caption_idx': caption_meta['caption_idx'],
            'score': float(f"{score:.4f}")
        })

# Save results
df = pd.DataFrame(all_search_results)
output_file = f'{file_name}_multilingual_results.csv'
df.to_csv(output_file, index=False)

print(f"Results saved to: {output_file}")
print(f"DataFrame shape: {df.shape}")
print(f"Language distribution:")
print(df['language'].value_counts())
print(f"Sample results:")
print(df.head(10))