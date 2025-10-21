from huggingface_hub import hf_hub_download
import pickle
import joblib
import os
import sys
import argparse # New import for command-line arguments

# Setting up a global constant for the default local cache directory
LOCAL_CACHE_DIR = os.path.join("data", "embeddings")

def _save_data_locally(data, filename, base_dir=LOCAL_CACHE_DIR):
    """
    Helper function to save the loaded data to a local cache path using pickle.
    
    Args:
        data (dict): The dictionary object to be saved.
        filename (str): The name of the file (used for the save path).
        base_dir (str): The base directory for caching.
    
    Returns:
        str: The full path where the data was saved.
    """
    model_name = filename[22:-12].replace('_', '-') 
    save_file_name = filename[:22] + model_name + filename[-12:]
    save_path = os.path.join(base_dir, save_file_name)
   
    try:
        # 1. Ensure the directory structure exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 2. Dump the data using pickle
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"Content successfully cached locally to: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"Warning: Failed to cache data locally to {save_path}. Error: {e}", file=sys.stderr)
        return None


def import_embeddings_huggingface_hub(repo_id: str, filename: str, save_dir: str = LOCAL_CACHE_DIR) -> dict or None:
    """
    Downloads a serialized dictionary file (embeddings, etc.) from the 
    Hugging Face Hub and attempts to load it using pickle, with a fallback to joblib. 
    If successful, the loaded content is cached locally using pickle.

    Args:
        repo_id (str): The ID of the dataset repository on the Hugging Face Hub.
        filename (str): The exact filename of the data object within the repository.
        save_dir (str): The local directory path to cache the downloaded file.

    Returns:
        dict or None: The loaded dictionary content, or None if an error occurred.
    """
    
    # 1. Input Check: Check if both required arguments are provided
    if not repo_id or not filename:
        print("Embedding repository ID or filename not provided. Skipping embedding import.")
        return None

    download_dict = None
    file_path = None

    try:
        # 2. Download the file from the Hugging Face Hub (Only happens once)
        print(f"Attempting to download '{filename}' from repo '{repo_id}'...")
        # Note: repo_type="dataset" is kept as per original code's intent
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        print(f"Download successful. File path: {file_path}")

        # 3. Unified Loading Logic: Try pickle, then fall back to joblib
        try:
            # Primary attempt: Load using the faster and native 'pickle'
            with open(file_path, 'rb') as f:
                download_dict = pickle.load(f)
            print("Successfully loaded content using 'pickle'.")
            
        except (pickle.UnpicklingError, EOFError) as e:
            # Fallback attempt: If pickle fails (often due to joblib serialization), try 'joblib'
            print(f"Pickle loading failed ({type(e).__name__}: {e}). Attempting fallback with 'joblib'...")
            
            with open(file_path, 'rb') as f:
                download_dict = joblib.load(f)
            print("Successfully loaded content using 'joblib'.")

        # 4. Save file to the local cache if loading was successful
        if download_dict is not None:
            # Pass the custom save_dir provided by the CLI (or the default)
            _save_data_locally(download_dict, filename, base_dir=save_dir)
            return download_dict
        
        return None # Should not be reached if exceptions are handled properly

    except Exception as e:
        # General error handling for download failure or joblib failure
        print("-" * 50, file=sys.stderr)
        print(f"FATAL ERROR during embedding import for '{filename}': {type(e).__name__} - {e}", file=sys.stderr)
        print("Please ensure the repository ID and filename are correct and accessible.", file=sys.stderr)
        print("-" * 50, file=sys.stderr)
        return None


def main():
    """
    Command-line entry point for the embedding import script.
    """
    parser = argparse.ArgumentParser(
        description="Download and cache serialized data (embeddings) from Hugging Face Hub."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="The ID of the dataset repository on the Hugging Face Hub (e.g., 'user/repo-name')."
    )
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="The exact filename of the data object within the repository (e.g., 'embeddings.pkl')."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=LOCAL_CACHE_DIR,
        help=f"The local directory path to cache the downloaded file. Default: '{LOCAL_CACHE_DIR}'"
    )
    
    args = parser.parse_args()

    result = import_embeddings_huggingface_hub(
        repo_id=args.repo_id, 
        filename=args.filename, 
        save_dir=args.save_dir
    )
    
    if result is not None:
        # We don't print the actual data (which could be huge)
        print("\nExecution complete. Data loaded and cached successfully.")
    else:
        print("\nExecution failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
