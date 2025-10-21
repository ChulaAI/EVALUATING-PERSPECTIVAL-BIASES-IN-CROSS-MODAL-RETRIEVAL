import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import json

class SelfPreferenceBiasEvaluator:
    def __init__(self, dataset, embeddings, model_name, model_type, processor=None):
        """
        Initialize BenchmarkEvaluator with dataset and model information.
        
        Args:
            dataset: CulturalBiasDataset object
            embeddings: Dictionary containing embeddings
            model_name: Name of the model
            model_type: Type of the model ('clip', 'siglip2', 'gme', 'colqwen', 'colpali')
            processor: Optional processor for late interaction models
        """
        self.dataset = dataset
        self.embeddings = embeddings
        self.model_name = model_name
        self.model_type = model_type
        self.processor = processor

    def calculate_matching_score(self, query_embedding, candidate_embedding):
        """
        Calculate matching score between query and candidate embeddings.
        """
        if self.model_type not in ['clip', 'siglip2', 'gme', 'colqwen2', 'colqwen2.5', 'colpali']:
            raise ValueError(f"Model type {self.model_type} not supported. Choose from 'clip', 'siglip2', 'gme', 'colqwen', 'colpali'.")

        if self.model_type in ['clip', 'siglip2', 'gme']:
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            score = cos(query_embedding, candidate_embedding)
        elif self.model_type in ['colqwen2', 'colqwen2.5', 'colpali']:
            score = self.processor.score_multi_vector(query_embedding, candidate_embedding)

        return score
    
    def equalize_benchmark_dataset_length(self, benchmark):
        """
        Equalize the length of the benchmark to match the dataset length by
        inserting entries for missing data.
        """
        dataset_length = len(self.dataset)
        benchmark_length = len(benchmark)

        if benchmark_length > dataset_length:
            raise ValueError("Benchmark length cannot be greater than dataset length.")
        
        # If lengths are already equal, no changes are needed.
        if benchmark_length == dataset_length:
            return benchmark
        
        new_benchmark = []
        current_benchmark_idx = 0

        # Iterate through the expected dataset length
        for i in range(dataset_length):
            # Check if the current benchmark item corresponds to the current dataset index
            if current_benchmark_idx < benchmark_length and benchmark[current_benchmark_idx]['query_index'] == i:
                new_benchmark.append(benchmark[current_benchmark_idx])
                current_benchmark_idx += 1
            else:
                # If a corresponding benchmark entry is missing, insert a placeholder
                new_benchmark.append({
                    "query_index": i,
                    "query_image_id": None,
                    "query_concept": None,
                    "query_country": None,
                    "semantically_relevant_candidate_index": None,
                    "culturally_relevant_candidate_index": None,
                    "non_relevant_candidate_index": None,
                    "semantically_relevant_candidate_image_id": None,
                    "culturally_relevant_candidate_image_id": None,
                    "non_relevant_candidate_image_id": None,
                    "semantically_relevant_candidate_concept": None,
                    "culturally_relevant_candidate_concept": None,
                    "non_relevant_candidate_concept": None,
                    "semantically_relevant_candidate_country": None,
                    "culturally_relevant_candidate_country": None,
                    "non_relevant_candidate_country": None
                })

        return new_benchmark

    def evaluate_test_case(self, test_case, experiment_type="text-to-image"):
        """
        Evaluate a single test case.
        """
        if experiment_type not in ["text-to-image", "image-to-image"]:
            raise ValueError("experiment_type must be either 'text-to-image' or 'image-to-image'")
            
        query_index = test_case['query_index']
        semantically_relevant_candidate_index = test_case['semantically_relevant_candidate_index']
        culturally_relevant_candidate_index = test_case['culturally_relevant_candidate_index']
        non_relevant_candidate_index = test_case['non_relevant_candidate_index']

        if semantically_relevant_candidate_index is None:
            return None, None

        if experiment_type == "image-to-image":
            text_query_embedding = self.embeddings['image_embedding'][query_index]
        else:  # text-to-image
            text_query_embedding = self.embeddings['text_embeddings']['translated_concept_embedding'][query_index]
            
        semantically_relevant_candidate_embedding = self.embeddings['image_embedding'][semantically_relevant_candidate_index]
        culturally_relevant_candidate_embedding = self.embeddings['image_embedding'][culturally_relevant_candidate_index]
        non_relevant_candidate_embedding = self.embeddings['image_embedding'][non_relevant_candidate_index]

        text_query_embedding = torch.from_numpy(np.array(text_query_embedding))
        semantically_relevant_candidate_embedding = torch.from_numpy(np.array(semantically_relevant_candidate_embedding))
        culturally_relevant_candidate_embedding = torch.from_numpy(np.array(culturally_relevant_candidate_embedding))
        non_relevant_candidate_embedding = torch.from_numpy(np.array(non_relevant_candidate_embedding))

        score_a = self.calculate_matching_score(text_query_embedding, semantically_relevant_candidate_embedding)
        score_b = self.calculate_matching_score(text_query_embedding, culturally_relevant_candidate_embedding)
        score_c = self.calculate_matching_score(text_query_embedding, non_relevant_candidate_embedding)
        score = torch.cat([score_a, score_b, score_c], 0)

        winner = ['a', 'b', 'c'][torch.argmax(score).item()]

        result = {
            "query_index": query_index,
            "query_image_id": test_case['query_image_id'],
            "query_concept": test_case['query_concept'],
            "query_country": test_case['query_country'],
            "semantically_relevant_candidate_index": semantically_relevant_candidate_index,
            "culturally_relevant_candidate_index": culturally_relevant_candidate_index,
            "non_relevant_candidate_index": non_relevant_candidate_index,    
            "semantically_relevant_candidate_image_id": test_case['semantically_relevant_candidate_image_id'],
            "culturally_relevant_candidate_image_id": test_case['culturally_relevant_candidate_image_id'],
            "non_relevant_candidate_image_id": test_case['non_relevant_candidate_image_id'],
            "semantically_relevant_candidate_concept": test_case['semantically_relevant_candidate_concept'],
            "culturally_relevant_candidate_concept": test_case['culturally_relevant_candidate_concept'],
            "non_relevant_candidate_concept": test_case['non_relevant_candidate_concept'],
            "semantically_relevant_candidate_country": test_case['semantically_relevant_candidate_country'],
            "culturally_relevant_candidate_country": test_case['culturally_relevant_candidate_country'],
            "non_relevant_candidate_country": test_case['non_relevant_candidate_country'],
            "score_a": float(score_a.reshape(-1)[0]),
            "score_b": float(score_b.reshape(-1)[0]),
            "score_c": float(score_c.reshape(-1)[0]),
            "winner": winner
        }

        return score, result

    def evaluate_from_benchmark(self, benchmark, experiment_type="text-to-image"):
        """
        Evaluate all test cases in the benchmark.
        """
        accuracy_list = []
        result_list = []

        benchmark = self.equalize_benchmark_dataset_length(benchmark)

        print('len new bench mark:', len(benchmark))

        for i in tqdm(range(len(benchmark))):
            score, result = self.evaluate_test_case(benchmark[i], experiment_type)
            if score is None:
                accuracy_list.append(None)
                result_list.append(None)
            else:
                accuracy_list.append(score.squeeze())
                result_list.append(result)
        return accuracy_list, result_list

    def summarize_scores(self, similarity_list):
        """
        Summarize scores from similarity list.
        """
        valid_scores = [s for s in similarity_list if s is not None]

        if len(valid_scores) == 0:
            raise ValueError("No valid similarity scores found!")

        scores_tensor = torch.stack(valid_scores).to(torch.float32)

        # One-hot max voting
        winners = torch.argmax(scores_tensor, dim=1)
        counts = torch.bincount(winners, minlength=3)
        one_hot_summary = {
            "A_wins": int(counts[0]),
            "B_wins": int(counts[1]),
            "C_wins": int(counts[2])
        }

        # Z-standardized pooling
        flat_scores = scores_tensor.view(-1).numpy()
        mean, std = flat_scores.mean(), flat_scores.std()
        z_scores = (scores_tensor.numpy() - mean) / std

        z_summary = {
            "A_avg_z": float(z_scores[:, 0].mean()),
            "B_avg_z": float(z_scores[:, 1].mean()),
            "C_avg_z": float(z_scores[:, 2].mean())
        }

        return one_hot_summary, z_summary

    def summarize_scores_with_metadata(self, similarity_list):
        """
        Summarize scores with metadata from the dataset.
        """
        df = self.dataset.to_pandas()

        # Add similarity scores to dataframe
        valid_scores = []
        for s in similarity_list:
            if s is None:
                valid_scores.append([None, None, None])
            else:
                s = s.to(torch.float32).tolist()
                valid_scores.append(s)

        df[["score_A", "score_B", "score_C"]] = pd.DataFrame(valid_scores, index=df.index)

        # Drop rows with None
        df = df.dropna(subset=["score_A", "score_B", "score_C"])

        # One-hot max voting
        df["winner"] = df[["score_A", "score_B", "score_C"]].values.argmax(axis=1)
        df["winner"] = df["winner"].map({0: "A", 1: "B", 2: "C"})

        # Z-standardization
        all_scores = df[["score_A", "score_B", "score_C"]].values.flatten()
        mean, std = all_scores.mean(), all_scores.std()
        for col in ["score_A", "score_B", "score_C"]:
            df[f"z_{col}"] = (df[col] - mean) / std

        # Summaries
        one_hot_summary = df["winner"].value_counts().to_dict()
        z_summary = df[["z_score_A", "z_score_B", "z_score_C"]].mean().to_dict()

        country_one_hot = df.groupby("country")["winner"].value_counts().unstack(fill_value=0)
        country_z = df.groupby("country")[["z_score_A", "z_score_B", "z_score_C"]].mean()

        concept_one_hot = df.groupby("concept")["winner"].value_counts().unstack(fill_value=0)
        concept_z = df.groupby("concept")[["z_score_A", "z_score_B", "z_score_C"]].mean()

        language_one_hot = df.groupby("native_language")["winner"].value_counts().unstack(fill_value=0)
        language_z = df.groupby("native_language")[["z_score_A", "z_score_B", "z_score_C"]].mean()

        return {
            "overall_one_hot": one_hot_summary,
            "overall_z": z_summary,
            "country_one_hot": country_one_hot,
            "country_z": country_z,
            "concept_one_hot": concept_one_hot,
            "concept_z": concept_z,
            "language_one_hot": language_one_hot,
            "language_z": language_z
        }

    def save_summaries(self, results, raw_results, experiment_type="text-to-image"):
        """
        Save evaluation summaries to files.
        """
        if experiment_type not in ["text-to-image", "image-to-image"]:
            raise ValueError("experiment_type must be either 'text-to-image' or 'image-to-image'")
        
        save_dir = os.path.join("evaluation_results", experiment_type)
        save_raw_dir = os.path.join("evaluation_results", "benchmark-results")
        os.makedirs(save_dir, exist_ok=True)

        # Overall summary
        overall_one_hot = pd.DataFrame([results["overall_one_hot"]]).add_suffix("_wins")
        overall_z = pd.DataFrame([results["overall_z"]]).rename(columns={
            "z_score_A": "z_A",
            "z_score_B": "z_B",
            "z_score_C": "z_C"
        })
        overall_summary = pd.concat([overall_one_hot, overall_z], axis=1)
        overall_summary.to_csv(os.path.join(save_dir, f"overall_summary_{self.model_name}.csv"), index=False)

        # Country summary
        country_summary = results["country_one_hot"].add_suffix("_wins").join(
            results["country_z"].rename(columns={
                "z_score_A": "z_A",
                "z_score_B": "z_B",
                "z_score_C": "z_C"
            }),
            how="outer"
        )
        country_summary.to_csv(os.path.join(save_dir, f"country_summary_{self.model_name}.csv"))

        # Concept summary
        concept_summary = results["concept_one_hot"].add_suffix("_wins").join(
            results["concept_z"].rename(columns={
                "z_score_A": "z_A",
                "z_score_B": "z_B",
                "z_score_C": "z_C"
            }),
            how="outer"
        )
        concept_summary.to_csv(os.path.join(save_dir, f"concept_summary_{self.model_name}.csv"))

        # Language summary
        language_summary = results["language_one_hot"].add_suffix("_wins").join(
            results["language_z"].rename(columns={
                "z_score_A": "z_A",
                "z_score_B": "z_B",
                "z_score_C": "z_C"
            }),
            how="outer"
        )
        language_summary.to_csv(os.path.join(save_dir, f"language_summary_{self.model_name}.csv"))

        with open(os.path.join(save_raw_dir, f"raw_evaluation_result_{self.model_name}.json"), "w") as f:
            json.dump(raw_results, f, indent=4) 

    def evaluate_and_save(self, benchmark, experiment_type="text-to-image"):
        """
        Convenience method to run the full evaluation pipeline.
        """
        accuracy_list, raw_results = self.evaluate_from_benchmark(benchmark, experiment_type)
        results = self.summarize_scores_with_metadata(accuracy_list)
        self.save_summaries(results, raw_results, experiment_type)
        return results