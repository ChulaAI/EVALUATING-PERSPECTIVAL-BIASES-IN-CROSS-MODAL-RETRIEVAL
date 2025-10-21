from huggingface_hub import hf_hub_download
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import pickle
import joblib
import os

def import_huggingface_hub(repo_id, filename):
    if repo_id is None or repo_id is None:
        print("Embedding repository ID or filename not provided. Skipping embedding import.")
        return None
    try:
      file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
      print(f"Downloaded file to: {file_path}")

      # 3. Load the pickle file
      with open(file_path, 'rb') as f:
          download_dict = pickle.load(f)
      return download_dict
    except Exception as e:
      try:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        print(f"Downloaded file to: {file_path}")

        # 3. Load the pickle file
        with open(file_path, 'rb') as f:
            download_dict = joblib.load(f)
        return download_dict
      except Exception as e:
        print(f"An error occurred during embedding import: {e}")
        print("Make sure the repository and file exist and are accessible.")
      return None

def import_local(dir_path, filename):
    file_path = f"{dir_path}/{filename}"
    try:
        with open(file_path, 'rb') as f:
            local_dict = pickle.load(f)
        return local_dict
    except Exception as e:
        try:
            with open(file_path, 'rb') as f:
                local_dict = joblib.load(f)
            return local_dict
        except Exception as e:
            print(f"An error occurred during local embedding import: {e}")
            print("Make sure the file exists and is accessible.")
        return None

class CulturalBiasDataset(Dataset):
    def __init__(self, path: str, embedding_repo_id: str = None, embedding_filename: str = None, split: str = 'train'):
        # Load the dataset from the specified path and split
        # dataset_dict = load_from_disk(path)

        dataset = load_dataset(path, split=split, name="image_metadata")
        self.image_id_to_index = None

        if embedding_repo_id and embedding_filename:
            self.embeddings = self.import_local_embedding(dir_path=embedding_repo_id, filename=embedding_filename)

            if self.embeddings:
                print('image_embedding:', len(self.embeddings['image_embedding']))
            if 'text_embeddings' in self.embeddings and 'translated_concept_embedding' in self.embeddings['text_embeddings']:
                print('caption_embedding:', len(self.embeddings['text_embeddings']['translated_concept_embedding']))
            if 'native_embedding' in self.embeddings:
                print('native_embedding:', len(self.embeddings['native_embedding']))

        # # Check if the loaded object is a DatasetDict and if the specified split exists
        # if isinstance(dataset_dict, DatasetDict) and split in dataset_dict:
        #      dataset = dataset_dict[split]
        # elif isinstance(dataset_dict, Dataset):
        #      # If load_dataset directly returned a Dataset (e.g., if split was specified in load_dataset)
        #      dataset = dataset_dict
        # else:
        #      raise TypeError(f"Expected Dataset or DatasetDict, but got {type(dataset_dict)}")

        # Access the specified split of the dataset
        dataset = self.add_native_language_column(dataset)

        # Initialize the underlying Arrow table
        super().__init__(dataset._data, info=dataset.info, split=dataset.split)

        # Create and store the image ID to index mapping
        self.image_id_to_index = self.create_image_id_to_index_map()


    # Add any custom methods or properties here
    def __getitem__(self, key):
        return super().__getitem__(key)

    def create_image_id_to_index_map(self):
        """
        Creates a dictionary that maps image IDs to their corresponding dataset indices.
        
        Returns:
            dict: A dictionary where keys are image IDs and values are their indices in the dataset
        """
        image_id_to_index = zip(self['image_id'], range(len(self)))
        image_id_to_index = dict(image_id_to_index)
        self.image_id_to_index = image_id_to_index
        return image_id_to_index

    def import_huggingface_hub_embedding(self, embedding_repo_id, embedding_filename):
      if embedding_repo_id is None or embedding_filename is None:
          print("Embedding repository ID or filename not provided. Skipping embedding import.")
          return None
      try:
        file_path = hf_hub_download(repo_id=embedding_repo_id, filename=embedding_filename, repo_type="dataset")
        print(f"Downloaded file to: {file_path}")

        # 3. Load the pickle file
        with open(file_path, 'rb') as f:
            embeddings_list_of_dicts = pickle.load(f)

        # Convert list of dictionaries to dictionary of lists
        if isinstance(embeddings_list_of_dicts, list) and all(isinstance(item, dict) for item in embeddings_list_of_dicts):
            embeddings_dict_of_lists = {}
            if embeddings_list_of_dicts: # Check if the list is not empty
                # Get all keys from the first dictionary (assuming all dicts have the same keys)
                keys = embeddings_list_of_dicts[0].keys()
                for key in keys:
                    # Check if the value is a nested dictionary (like 'text_embeddings')
                    if key == 'text_embeddings':
                        embeddings_dict_of_lists[key] = {}
                        if embeddings_list_of_dicts[0].get(key) and isinstance(embeddings_list_of_dicts[0][key], dict):
                             inner_keys = embeddings_list_of_dicts[0][key].keys()
                             for inner_key in inner_keys:
                                 embeddings_dict_of_lists[key][inner_key] = [d[key][inner_key] for d in embeddings_list_of_dicts if key in d and inner_key in d[key]]
                    else:
                        embeddings_dict_of_lists[key] = [d[key] for d in embeddings_list_of_dicts]
            print("Converted embeddings to dictionary of lists.")
            return embeddings_dict_of_lists
        else:
            print("Loaded embeddings are not in the expected list of dictionaries format. Returning loaded data directly.")
            return embeddings_list_of_dicts # Return as is if not list of dicts

      except Exception as e:
        print(f"An error occurred during embedding import: {e}")
        print("Make sure the repository and file exist and are accessible.")
        return None
    
    def import_local_embedding(self, dir_path, filename):
        file_path = os.path.join(dir_path, filename)
        print('file path:', file_path)
        try:
            with open(file_path, 'rb') as f:
                embeddings_list_of_dicts = pickle.load(f)
            
        except Exception as e:
            try:
                with open(file_path, 'rb') as f:
                    embeddings_list_of_dicts = joblib.load(f)
        
            except Exception as e:
                print(f"An error occurred during local embedding import: {e}")
                print("Make sure the file exists and is accessible.")

        try:
            # Convert list of dictionaries to dictionary of lists
            if isinstance(embeddings_list_of_dicts, list) and all(isinstance(item, dict) for item in embeddings_list_of_dicts):
                embeddings_dict_of_lists = {}
                if embeddings_list_of_dicts: # Check if the list is not empty
                    # Get all keys from the first dictionary (assuming all dicts have the same keys)
                    keys = embeddings_list_of_dicts[0].keys()
                    for key in keys:
                        # Check if the value is a nested dictionary (like 'text_embeddings')
                        if key == 'text_embeddings':
                            embeddings_dict_of_lists[key] = {}
                            if embeddings_list_of_dicts[0].get(key) and isinstance(embeddings_list_of_dicts[0][key], dict):
                                inner_keys = embeddings_list_of_dicts[0][key].keys()
                                for inner_key in inner_keys:
                                    embeddings_dict_of_lists[key][inner_key] = [d[key][inner_key] for d in embeddings_list_of_dicts if key in d and inner_key in d[key]]
                        else:
                            embeddings_dict_of_lists[key] = [d[key] for d in embeddings_list_of_dicts]
                print("Converted embeddings to dictionary of lists.")
                self.embeddings = embeddings_dict_of_lists
                return embeddings_dict_of_lists
            else:
                print("Loaded embeddings are not in the expected list of dictionaries format. Returning loaded data directly.")
                return embeddings_list_of_dicts # Return as is if not list of dicts

        except Exception as e:
            print(f"An error occurred during embedding import: {e}")
            print("Make sure the repository and file exist and are accessible.")
            return None

    def get_image_embeddings(self, ids):
        if self.embeddings and 'image_embedding' in self.embeddings:
            if isinstance(ids, list):
                # Return the original structure
                return [self.embeddings['image_embedding'][i] for i in ids]
            else:
                 # Return the original structure
                return self.embeddings['image_embedding'][ids]
        return None

    def get_concept_embeddings(self, ids):
        if self.embeddings and 'text_embeddings' in self.embeddings and 'concept_embedding' in self.embeddings['text_embeddings']:
            if isinstance(ids, list):
                 # Return the original structure
                return [self.embeddings['text_embeddings']['concept_embedding'][i] for i in ids]
            else:
                 # Return the original structure
                return self.embeddings['text_embeddings']['concept_embedding'][ids]
        return None

    def get_translated_concept_embeddings(self, ids):
      if self.embeddings and 'text_embeddings' in self.embeddings and 'translated_concept_embedding' in self.embeddings['text_embeddings']:
          if isinstance(ids, list):
                # Return the original structure
              return [self.embeddings['text_embeddings']['translated_concept_embedding'][i] for i in ids]
          else:
                # Return the original structure
              return self.embeddings['text_embeddings']['translated_concept_embedding'][ids]
      return None


    def add_native_language_column(self, dataset):
      country_to_language = {
        "Argentina": "Spanish",
        "Australia": "English",
        "Brazil": "Portuguese",
        "China": "Chinese",
        "France": "French",
        "Germany": "German",
        "India": "Hindi",
        "Japan": "Japanese",
        "Kenya": "Swahili",
        "Nigeria": "Yoruba",
        "Portugal": "Portuguese",
        "Saudi Arabia": "Arabic",
        "Spain": "Spanish",
        "Thailand": "Thai",
        "UK": "English",
        "USA": "English"
        }
      self.country_to_language = country_to_language
      def add_language(example):
          country = example['country']
          example['native_language'] = country_to_language.get(country, None) # Use .get for safer access
          return example

      # Use the map method to add the new column
      dataset = dataset.map(add_language)
      return dataset
