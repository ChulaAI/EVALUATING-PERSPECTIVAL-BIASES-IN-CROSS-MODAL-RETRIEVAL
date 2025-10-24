# Evaluating Perspectival Biases in Cross-Modal Retrieval

This repository contains the implementation and evaluation framework for analyzing perspectival biases in cross-modal retrieval systems. The project investigates how different embedding models perform across various cultural contexts, languages, and demographic groups in image-text retrieval tasks.

## 🔍 Overview

Cross-modal retrieval systems often exhibit biases that reflect the perspectives and cultural contexts of their training data. This project provides a comprehensive framework to evaluate and measure these biases across multiple dimensions:

- **Cultural Bias**: How models perform across different cultural contexts and countries
- **Language Bias**: Performance variations across different languages and scripts
- **Demographic Bias**: Differences in model behavior across various demographic groups
- **Self-Preference Bias**: Tendency of models to favor content from their training distribution

## 📊 Research Questions

### RQ1: Cross-Modal Performance Evaluation
Evaluates how different embedding models perform in cross-modal retrieval tasks using the Crossmodal-3600 dataset, focusing on:
- Image-to-text retrieval accuracy
- Text-to-image retrieval performance  
- Language-specific performance variations
- Model comparison across different architectures

### RQ2: Cultural and Demographic Bias Analysis
Analyzes perspectival biases using the Association Bias Benchmark, examining:
- Self-preference tendencies in embedding models
- Cultural representation disparities
- Concept association biases across different regions
- Model fairness across demographic groups

## 🏗️ Project Structure

```
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore patterns
│
├── rq1_eval/                          # RQ1 Cross-Modal Evaluation
│   ├── 1_embedding.ipynb              # Interactive embedding generation
│   ├── 2A_retrieve_normal.ipynb       # Standard retrieval evaluation
│   ├── 2B_retrieve_col.ipynb          # ColQwen retrieval evaluation
│   ├── 3_viz_imgtotext.ipynb          # Image-to-text visualization
│   ├── 3_viz_lang_frequency.ipynb     # Language frequency analysis
│   ├── embeddings/                    # Generated embeddings storage
│   └── results_crossmodal/            # Evaluation results
│
├── rq2_eval/                          # RQ2 Bias Analysis
│   ├── 1_embedding.ipynb              # Interactive embedding generation
│   ├── self_preference_evaluation.ipynb # Main bias evaluation
│   ├── self_preference_analyze_*.ipynb # Analysis notebooks
│   ├── single_modal_bias.ipynb        # Single-modal bias analysis
│   ├── benchmarks/                    # Benchmark datasets
│   ├── embeddings/                    # Generated embeddings storage
│   ├── evaluation_results/            # Bias evaluation results
│   ├── fonts/                         # Fonts for visualization
│   └── utils/                         # Utility functions
│
└── scripts/                           # Utility scripts
    └── load_embedding.py              # Embedding loading utilities
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher (Python 3.13 recommended)
- CUDA-capable GPU (recommended for faster processing)
- At least 16GB RAM
- 50GB+ free disk space for embeddings and results

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/TierOne2025/EVALUATING-PERSPECTIVAL-BIASES-IN-CROSS-MODAL-RETRIEVAL.git
cd EVALUATING-PERSPECTIVAL-BIASES-IN-CROSS-MODAL-RETRIEVAL
```

2. **Create and activate Python virtual environment**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate environment (Linux/Mac)
source .venv/bin/activate

# Activate environment (Windows)
.venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up additional dependencies** (if needed)
```bash
# For ColQwen models
pip install colpali-engine

# For multilingual CLIP
pip install multilingual-clip

# For OpenCLIP
pip install open-clip-torch
```

### Configuration

Set up your HuggingFace token for accessing gated models:
```bash
export HF_TOKEN="your_huggingface_token_here"
```

## 📝 Usage Guide

### Step 1: Generate Embeddings

#### For RQ1 (Cross-Modal Evaluation):
```bash
# Open the interactive notebook
jupyter notebook rq1_eval/1_embedding.ipynb
```

**Supported Models:**
- GME (Qwen2-VL variants)
- ColQwen (ColQwen2, ColQwen2.5)
- Jina (jina-embeddings-v4)
- Multilingual CLIP (XLM-Roberta variants)

#### For RQ2 (Bias Analysis):
```bash
# Open the interactive notebook  
jupyter notebook rq2_eval/1_embedding.ipynb
```

**Supported Models:**
- CLIP (OpenAI CLIP, Chinese CLIP)
- ColQwen (ColQwen2, ColQwen2.5)
- GME (Qwen2-VL variants)
- Jina (jina-embeddings-v4)
- Multilingual CLIP (XLM-Roberta variants)

### Step 2: Run Evaluations

#### RQ1 Evaluation:
```bash
# Standard retrieval evaluation
jupyter notebook rq1_eval/2A_retrieve_normal.ipynb

# ColQwen-specific evaluation
jupyter notebook rq1_eval/2B_retrieve_col.ipynb

# Generate visualizations
jupyter notebook rq1_eval/3_viz_imgtotext.ipynb
jupyter notebook rq1_eval/3_viz_lang_frequency.ipynb
```

#### RQ2 Evaluation:
```bash
# Main bias evaluation
jupyter notebook rq2_eval/self_preference_evaluation.ipynb

# Detailed analysis
jupyter notebook rq2_eval/self_preference_analyze_overall.ipynb
jupyter notebook rq2_eval/self_preference_analyze_concept.ipynb
jupyter notebook rq2_eval/self_preference_analyze_country.ipynb
jupyter notebook rq2_eval/self_preference_analyze_language.ipynb
```

## 🔧 Advanced Configuration

### Memory Management
For large-scale processing, you can configure memory-efficient settings:

```python
# In embedding notebooks
START_INDEX = 0        # Start processing from this index
END_INDEX = 1000       # Process up to this index (None = all)
BATCH_SIZE = 32        # Adjust based on GPU memory
```

### Model Selection
Each notebook provides interactive widgets to select models:

- **Dropdown menus** for easy model switching
- **Real-time configuration** updates
- **Automatic model loading** and setup

### Output Configuration
Customize output paths and formats:

```python
# Set custom paths
EMBEDDING_SAVE_PATH = "custom_embeddings"
RESULTS_SAVE_PATH = "custom_results" 
```

## 📊 Datasets

### RQ1: Crossmodal-3600
- **Purpose**: Multilingual image-text retrieval evaluation
- **Size**: 3,600 images with captions in 36 languages
- **Usage**: Cross-modal retrieval performance analysis

### RQ2: Association Bias Benchmark
- **Purpose**: Cultural and demographic bias evaluation
- **Dataset**: `Chula-AI/association_bias_benchmark`
- **Features**: Cultural concepts across different countries and languages
- **Usage**: Bias measurement and fairness analysis

## 📈 Output Formats

### Embedding Files
```python
# Pickle format with structured data
{
    'image_key': str,              # RQ1: Image identifier
    'concept_id': str,             # RQ2: Concept identifier  
    'image_embedding': np.ndarray,  # Image embedding vector
    'text_embeddings': {           # Text embeddings dictionary
        'caption_embedding_en': np.ndarray,     # RQ1: English captions
        'concept_embedding': np.ndarray,        # RQ2: Main concept
        'translated_concept_embedding': np.ndarray  # RQ2: Native language
    }
}
```

### Evaluation Results
- **CSV files** with detailed metrics
- **Aggregated summaries** across models and conditions
- **Visualization-ready** data formats

## 🔬 Evaluation Metrics

### RQ1 Metrics:
- **Recall@K** (K=1,5,10): Retrieval accuracy at different cutoffs
- **Mean Reciprocal Rank (MRR)**: Average ranking performance
- **Language-specific performance**: Per-language breakdown

### RQ2 Metrics:
- **Self-preference scores**: Bias towards training distribution
- **Cultural fairness**: Performance equality across cultures
- **Concept association bias**: Stereotype measurement
- **Demographic parity**: Fairness across demographic groups

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or process in chunks
   START_INDEX = 0
   END_INDEX = 500  # Process in smaller batches
   ```

2. **Model Loading Errors**
   ```bash
   # Ensure HuggingFace token is set
   export HF_TOKEN="your_token"
   
   # Install missing dependencies
   pip install transformers[torch]
   ```

3. **Dataset Loading Issues**
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/huggingface/datasets/
   ```

### Performance Optimization

- **Use CUDA**: Ensure PyTorch detects your GPU
- **Batch processing**: Process embeddings in manageable chunks  
- **Memory monitoring**: Monitor GPU memory usage
- **Parallel processing**: Use multiple GPUs if available

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{perspectival-bias-eval-2024,
    title={Evaluating Perspectival Biases in Cross-Modal Retrieval},
    author={[Authors]},
    year={2024},
    howpublished={\url{https://github.com/TierOne2025/EVALUATING-PERSPECTIVAL-BIASES-IN-CROSS-MODAL-RETRIEVAL}}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- HuggingFace for providing model hosting and datasets
- The authors of Crossmodal-3600 and Association Bias Benchmark datasets
- The open-source community for the various embedding models used

## 📞 Contact

For questions or issues, please:
- Open an issue in this repository

---

**Happy researching! 🚀**

