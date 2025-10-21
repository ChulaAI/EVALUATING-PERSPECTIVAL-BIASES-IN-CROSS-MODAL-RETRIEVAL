# Import libraries
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties, fontManager
import matplotlib as mpl
from typing import Dict, List, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import dataset loader
from .load_cultural_dataset import CulturalBiasDataset

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


# Configuration
class Config:
    """Configuration class for paths and constants."""
    AGGREGATED_RESULTS_PATH = os.path.join('evaluation_results', 'aggregated')
    RAW_EVALUATION_PATH = os.path.join('evaluation_results', 'benchmark-results')
    DATASET_PATH = os.path.join('datasets', 'tierone003_deduplicated_and_renamed')
    FONT_PATH = os.path.join('fonts', 'tahoma.ttf')
    
    # Ordered lists for consistent plotting
    SORTED_COUNTRIES_BY_RESOURCE = [
        "USA", "UK", "Australia", "Germany", "China", 
        "Japan", "France", "Spain", "Argentina", "Portugal", "Brazil", 
        "Saudi Arabia", "Thailand", "India", "Kenya", "Nigeria"
    ]
    
    SORTED_LANGUAGES_BY_RESOURCE = [
        "English", "German", "Chinese", "Japanese", "French", 
        "Spanish", "Portuguese", "Arabic", "Thai", "Hindi", "Swahili", "Yoruba"
    ]
    
    # Column mappings
    COLUMN_MAPPING = {
        "A_wins": "Conceptual Relevant Wins",
        "B_wins": "Cultural Relevant Wins", 
        "C_wins": "Non-Relevant Wins"
    }
    
    # Win types for consistent labeling
    WIN_TYPES = ["Conceptual Relevant Wins", "Cultural Relevant Wins", "Non-Relevant Wins"]
    
    # Plot styling
    FIGURE_SIZE = (15, 7)
    LARGE_FIGURE_SIZE = (25, 15)
    BAR_WIDTH = 0.25
    

class AnalyzeDataLoader:
    """Class for loading and preprocessing evaluation data."""
    
    def __init__(self, config: Config):
        self.config = config
        self.dataframes = {}
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all evaluation result DataFrames."""
        file_mappings = {
            'concept': 'concept_aggregated.csv',
            'country': 'country_aggregated.csv', 
            'overall': 'overall_aggregated.csv',
            'language': 'language_aggregated.csv'
        }
        
        for key, filename in file_mappings.items():
            filepath = os.path.join(self.config.AGGREGATED_RESULTS_PATH, filename)
            self.dataframes[key] = pd.read_csv(filepath)
            
        return self.dataframes
    
    def rename_columns(self) -> None:
        """Rename columns to more descriptive names."""
        for df in self.dataframes.values():
            df.rename(columns=self.config.COLUMN_MAPPING, inplace=True)
    
    def prepare_data(self) -> Dict[str, pd.DataFrame]:
        """Load and preprocess all data."""
        self.load_data()
        self.rename_columns()
        return self.dataframes


class PlotUtils:
    """Utility class for common plotting operations."""
    
    @staticmethod
    def calculate_percentages(df: pd.DataFrame, win_types: List[str]) -> pd.DataFrame:
        """Calculate win percentages for a DataFrame."""
        df = df.copy()
        df['Total_wins'] = df[win_types].sum(axis=1)
        
        for win_type in win_types:
            percentage_col = f'{win_type}_percentage'
            df[percentage_col] = (df[win_type] / df['Total_wins'] * 100).fillna(0)
        
        return df
    
    @staticmethod
    def add_data_labels(ax, bars, rotation: int = 0):
        """Add percentage labels to bars."""
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{height:.1f}%', ha='center', va='bottom', 
                       rotation=rotation, fontsize=40)
    
    @staticmethod
    def setup_bar_plot(ax, data: pd.DataFrame, title: str, xlabel: str, 
                    win_types: List[str], bar_width: float = 0.25, datalabel_rotaion=0):
        """Set up a standard bar plot with win types."""
        index = np.arange(len(data))
        bars = []
        
        for i, win_type in enumerate(win_types):
            percentage_col = f'{win_type}_percentage'
            bar = ax.bar(index + i * bar_width, data[percentage_col], 
                        bar_width, label=win_type)
            bars.append(bar)
            PlotUtils.add_data_labels(ax, bar, rotation=datalabel_rotaion) 
        
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel('Percentage of Wins', fontsize=15)
        ax.set_title(title, fontsize=17)
        ax.set_xticks(index + bar_width)
        ax.legend(loc='upper right', fontsize=10) # Changed this line
        ax.grid(axis='y', alpha=0.3)
        
        # Increase tick label font size
        ax.tick_params(axis='both', which='major', labelsize=10)

        return bars
    
    @staticmethod
    def setup_stacked_bar_plot(ax, data: pd.DataFrame, title: str, xlabel: str, 
                            win_types: List[str], datalabel_rotation=0, show_data_labels=True, xlabel_fontsize=19):
        """Set up a stacked bar plot with win types."""
        colors = ['#ff9999', '#b8860b', '#90ee90']  # Light red, dark goldenrod, light green
        bottom = None
        
        for i, win_type in enumerate(win_types):
            percentage_col = f'{win_type}_percentage'
            bars = ax.bar(range(len(data)), data[percentage_col], 
                         bottom=bottom, label=win_type, color=colors[i])
            
            # Add percentage labels only for segments larger than 5%
            if show_data_labels == True:
                for j, (index_name, row) in enumerate(data.iterrows()):
                    value = row[percentage_col]
                    if value > 5:  # Only show label if segment is large enough
                        y_pos = (bottom[j] if bottom is not None else 0) + value/2
                        ax.text(j, y_pos, f'{value:.1f}%', ha='center', va='center', 
                                fontsize=20, color='black')
            
            # Update bottom for next stack
            if bottom is None:
                bottom = data[percentage_col].values
            else:
                bottom += data[percentage_col].values
        
        # Customize plot
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel('Percentage of Wins', fontsize=15)
        ax.set_title(title, fontsize=17)
        ax.set_xticks(range(len(data)))
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_ylim(0, 100)
        
        return bars


class QualitativeEvaluator:
    """Class for qualitative evaluation and visualization of raw results."""
    
    def __init__(self, config: Config):
        self.config = config
        self.font_prop = self._setup_font()
        self.dataset = None
        self.image_id_to_index = None
    
    def _setup_font(self) -> FontProperties:
        """Set up multilingual font for text display."""
        current_path = os.getcwd()
        font_path = os.path.join(current_path, self.config.FONT_PATH)
        
        if os.path.exists(font_path):
            print(f"Loading font from: {font_path}")
            font_prop = FontProperties(fname=font_path)
            
            # Configure matplotlib to use our font globally
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font_prop.get_name()] + plt.rcParams['font.sans-serif']
            print("Font configuration complete.")
            return font_prop
        else:
            print(f"Warning: Font file not found at {font_path}, using default font")
            return FontProperties(family='sans-serif')
    
    def load_raw_results(self, file_path: str) -> pd.DataFrame:
        """Load raw evaluation results for a specific model."""
        
        try:
            with open(file_path, 'r') as f:
                raw_results = json.load(f)
            return pd.json_normalize(raw_results)
        except FileNotFoundError:
            raise FileNotFoundError(f"Raw evaluation file not found: {file_path}")

    def load_dataset(self, dataset_path: Optional[str] = None) -> None:
        """Load the cultural bias dataset."""
        if dataset_path is None:
            dataset_path = self.config.DATASET_PATH
        
        self.dataset = CulturalBiasDataset(dataset_path)
        self.image_id_to_index = self.dataset.image_id_to_index
        print(f"Dataset loaded from: {dataset_path}")
    
    def show_sample_raw_results(self, df: pd.DataFrame, 
                               winner: Optional[str] = None, 
                               query_concept: Optional[str] = None, 
                               semantically_relevant_candidate_concept: Optional[str] = None,
                               culturally_relevant_candidate_concept: Optional[str] = None,
                               non_relevant_candidate_concept: Optional[str] = None,
                               sample_size: int = 5) -> pd.DataFrame:
        """
        Show sample raw results filtered by winner and/or concept, including images and information as plot.
        
        Parameters:
        df (pd.DataFrame): Raw result dataframe
        winner (str): Filter by winner ('a', 'b', or 'c')
        query_concept (str): Filter by query concept
        semantically_relevant_candidate_concept (str): Filter by candidate A concept
        culturally_relevant_candidate_concept (str): Filter by candidate B concept  
        non_relevant_candidate_concept (str): Filter by candidate C concept
        sample_size (int): Number of samples to return
        
        Returns:
        pd.DataFrame: Filtered sample dataframe
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # Apply filters
        filtered_df = df.copy()
        
        if winner:
            filtered_df = filtered_df[filtered_df['winner'] == winner]
        
        if query_concept:
            filtered_df = filtered_df[filtered_df['query_concept'] == query_concept]
        
        if semantically_relevant_candidate_concept:
            filtered_df = filtered_df[filtered_df['semantically_relevant_candidate_concept'] == semantically_relevant_candidate_concept]
        
        if culturally_relevant_candidate_concept:
            filtered_df = filtered_df[filtered_df['culturally_relevant_candidate_concept'] == culturally_relevant_candidate_concept]
        
        if non_relevant_candidate_concept:
            filtered_df = filtered_df[filtered_df['non_relevant_candidate_concept'] == non_relevant_candidate_concept]
        
        # Get random sample
        sample_df = filtered_df.sample(n=min(sample_size, len(filtered_df)))
        
        winner_to_idx = {'a': 0, 'b': 1, 'c': 2}
        
        # Display results with images
        for _, row in sample_df.iterrows():
            winner_idx = winner_to_idx[row['winner']]
            
            # Create figure and grid layout
            fig = plt.figure(figsize=(15, 8))
            gs = plt.GridSpec(2, 3, height_ratios=[1, 3], figure=fig)
            
            # Top row for text information (spans all columns)
            ax_text = fig.add_subplot(gs[0, :])
            ax_text.axis('off')
            
            # Get query information with proper encoding
            query_idx = self.image_id_to_index[row['query_image_id']]
            query_native = self.dataset[query_idx]['concept_in_native']
            query_country = self.dataset[query_idx]['country']
            query_translation = self.dataset[query_idx]['concept']
            query_language = self.dataset.country_to_language.get(query_country, 'Unknown')
            
            # Format text information with proper line breaks and spacing
            query_info = [
                f"Query {int(row['query_index'])}",
                f"Native Text [{query_language}]: {query_native}",
                f"Country: {query_country} | Translation: {query_translation}",
                f"Scores: A={row['score_a']:.3f}, B={row['score_b']:.3f}, C={row['score_c']:.3f}"
            ]
            
            # Add text box with query information using the multilingual font
            ax_text.text(0.5, 0.5, '\n'.join(query_info),
                        ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8,
                                boxstyle='round,pad=0.5'),
                        transform=ax_text.transAxes,
                        fontsize=10,
                        fontproperties=self.font_prop,
                        linespacing=1.5)
            
            # Bottom row for images (three columns)
            titles = ['Candidate A', 'Candidate B', 'Candidate C']
            image_ids = [row['semantically_relevant_candidate_image_id'], row['culturally_relevant_candidate_image_id'], row['non_relevant_candidate_image_id']]
            
            for i, image_id in enumerate(image_ids):
                ax = fig.add_subplot(gs[1, i])
                if image_id in self.image_id_to_index:
                    idx = self.image_id_to_index[image_id]
                    image = self.dataset[idx]['image']
                    ax.imshow(image)
                    
                    # Get the language for this image's country
                    img_country = self.dataset[idx]['country']
                    img_concept = self.dataset[idx]['concept']
                    img_language = self.dataset.country_to_language.get(img_country, 'Unknown')
                    
                    # Add image ID and language below the image
                    ax.text(0.5, -0.1, f"ID: {image_id}\nCountry: {img_country}\nLanguage: {img_language}\nConcept: {img_concept}", 
                           ha='center', va='top', 
                           transform=ax.transAxes,
                           fontsize=8,
                           fontproperties=self.font_prop,
                           linespacing=1.2)
                    
                    if i == winner_idx:
                        title_color = 'green'
                        # Add winner border
                        for spine in ax.spines.values():
                            spine.set_edgecolor('green')
                            spine.set_linewidth(2)
                    else:
                        title_color = 'black'
                    
                    ax.set_title(titles[i], color=title_color, pad=10, fontproperties=self.font_prop)
                else:
                    ax.text(0.5, 0.5, 'Image not found', 
                           ha='center', va='center', 
                           fontproperties=self.font_prop)
                
                ax.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return sample_df