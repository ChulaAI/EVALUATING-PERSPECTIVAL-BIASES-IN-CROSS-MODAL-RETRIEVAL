#!/bin/bash

SCRIPT_TO_RUN="scripts/load_embedding.py"

# --- EMBEDDING PARAMETERS ---
# RQ1 parameters
RQ1_SAVE_DIRECTORY="rq1_eval/embeddings/."
RQ1_EMBEDDING_HF_REPO="tierone004/embedding_rq3_part_rq1_redo" 
RQ1_EMBEDDING_RQ1_TRIPLETS=(
    "$RQ1_EMBEDDING_HF_REPO image_text_embeddings_XLM-Roberta-Large-Vit-B-16Plus_0_3600.pkl clip"
    "$RQ1_EMBEDDING_HF_REPO image_text_embeddings_XLM-Roberta-Large-Vit-L-14_0_3600.pkl clip"
    "$RQ1_EMBEDDING_HF_REPO image_text_embeddings_gme2B_0_3600.pkl gme"
    "$RQ1_EMBEDDING_HF_REPO image_text_embeddings_gme7B_0_3600.pkl gme"
    "$RQ1_EMBEDDING_HF_REPO image_text_embeddings_jina-embeddings-v4_0_3600.pkl clip"
)

# RQ2 parameters
RQ2_SAVE_DIRECTORY="rq2_eval/embeddings/."
RQ2_EMBEDDING_HF_REPO="tierone004/redo_embedding_rq2_fix_translation" 
RQ2_EMBEDDING_RQ2_TRIPLETS=( # Corrected array name for consistency
    "$RQ2_EMBEDDING_HF_REPO image_text_embeddings_clip_vit_large_patch14_0_11759.pkl clip"
    "$RQ2_EMBEDDING_HF_REPO image_text_embeddings_jina-embeddings-v4_0_11759.pkl jina"
    "$RQ2_EMBEDDING_HF_REPO image_text_embeddings_XLM-Roberta-Large-Vit-L-14_0_11759.pkl clip"
    "$RQ2_EMBEDDING_HF_REPO image_text_embeddings_XLM-Roberta-Large-Vit-B-16Plus_0_11759.pkl clip"
    "$RQ2_EMBEDDING_HF_REPO image_text_embeddings_colqwen2.5-v0.2_0_11759.pkl colqwen2.5_v02"
    "$RQ2_EMBEDDING_HF_REPO image_text_embeddings_gme-Qwen2-VL-2B-Instruct_0_11759.pkl gme"
    "$RQ2_EMBEDDING_HF_REPO image_text_embeddings_ColQwen2.5-3b-multilingual-v1.0_0_11759.pkl colqwen2.5_3b"
)

# =========================================================================

## Function Definitions

# Function to execute the embedding import loop
# Arguments:
#   $1: Name of the array holding the triplets (e.g., RQ1_EMBEDDING_RQ1_TRIPLETS)
#   $2: The target save directory (e.g., rq1_eval/embeddings/.)
run_embedding_import() {
    local ARRAY_NAME="$1[@]"
    local SAVE_DIR="$2"
    
    echo ""
    echo "--- Starting Embedding Import for: ${ARRAY_NAME} (Saving to: $SAVE_DIR) ---"

    # Use indirect expansion to access the elements of the passed array
    for triplet in "${!ARRAY_NAME}"; do
        # Read the triplet into variables
        read -r REPO_ID FILENAME MODEL_NAME <<< "$triplet"
        
        if [ -z "$MODEL_NAME" ]; then
            echo "❌ ERROR: Skipping item. Failed to read all three components from: '$triplet'"
            continue
        fi
        
        echo ""
        echo "➡️ Processing: REPO_ID=$REPO_ID, FILENAME=$FILENAME, MODEL_NAME=$MODEL_NAME"
        
        # Construct and execute the command
        EMBEDDING_IMPORTER_COMMAND="python $SCRIPT_TO_RUN \
            --repo_id $REPO_ID \
            --filename $FILENAME \
            --save_dir $SAVE_DIR" 
        
        echo "Executing command: $EMBEDDING_IMPORTER_COMMAND"
        
        # Run the command
        $EMBEDDING_IMPORTER_COMMAND
        
        # Check the exit status of the python script
        if [ $? -eq 0 ]; then
            echo "✅ Success: $REPO_ID ($MODEL_NAME) processed."
        else
            echo "❌ Failure: $REPO_ID ($MODEL_NAME) failed during processing."
        fi
    done
}

# --------------------------------------------------------------------------
## 🚀 RUN THE FUNCTIONS
# --------------------------------------------------------------------------

# Run the import for RQ1 data
# run_embedding_import "RQ1_EMBEDDING_RQ1_TRIPLETS" "$RQ1_SAVE_DIRECTORY"

# Run the import for RQ2 data
run_embedding_import "RQ2_EMBEDDING_RQ2_TRIPLETS" "$RQ2_SAVE_DIRECTORY"


# --------------------------------------------------------------------------
## ✨ Cleanup

# Deactivate the virtual environment
deactivate
echo ""
echo "Virtual environment deactivated."
echo "Script finished. All embedding imports complete."