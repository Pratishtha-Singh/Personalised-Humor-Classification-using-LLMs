# Personalised Humor Classification using LLMs

A comprehensive machine learning project for personalised humor classification that predicts how individuals perceive jokes based on demographic context. The system classifies jokes into three categories: **funny**, **not_funny**, or **don't_understand**, considering user demographics (age, gender, ethnicity).

## Project Overview

This project implements multiple approaches to humor classification, from traditional machine learning baselines to advanced transformer models and large language models (LLMs). The goal is to create a personalized humor recommendation system that understands individual preferences and cultural contexts.

### Key Features

- **Multi-label Classification**: Three-way classification (funny/not_funny/don't_understand)
- **Demographic-aware**: Incorporates user age, gender, and ethnicity
- **Cross-cultural Dataset**: Diverse joke collection with user responses
- **Multiple Model Architectures**: From TF-IDF to transformer models
- **Calibrated Predictions**: Uncertainty-aware outputs
- **Interactive Dashboard**: Streamlit-based visualization and testing interface

## Dataset

The project uses a curated humor dataset containing:
- **Joke texts** from various sources and cultures
- **User demographics** (age groups, gender, ethnicity)
- **Response labels** (funny/not_funny/don't_understand)
- **Cross-cultural representation** for diverse humor understanding

## Model Architectures

### 1. Baseline Models
- **TF-IDF + Logistic Regression**: Traditional NLP approach
- **DistilBERT**: Pre-trained transformer baseline

### 2. Novelty Model
- **RoBERTa + Demographics**: Fine-tuned with LoRA (Low-Rank Adaptation)
- **Demographic Integration**: Age, gender, and ethnicity embeddings
- **Cost-sensitive Training**: Handles class imbalance

### 3. LLM Approach
- **Gemini Integration**: Using Google's Gemini model for few-shot classification
- **Prompt Engineering**: Carefully crafted prompts for demographic-aware humor understanding

### 4. Meta Pipeline
- **Two-stage Classification**: Combines multiple model predictions
- **Calibrated Outputs**: Uncertainty quantification
- **Ensemble Methods**: Leverages strengths of different approaches

## Repository Structure

```
├── base1_model_building.ipynb          # Baseline model development
├── base2_model_building.ipynb          # Advanced baseline models
├── Gemini_model.ipynb                  # LLM-based classification
├── novelty_model.ipynb                 # Novel RoBERTa approach
├── Project_File.ipynb                  # Main project notebook
├── streamlit_dashboard.ipynb           # Interactive dashboard
├── humor_dashboard.py                  # Dashboard implementation
├── Dataset.csv                         # Original dataset
├── humor_modeling_dataset.csv          # Processed dataset
├── preprocessed_humor_data.csv         # Final preprocessed data
├── humor_model_artifacts/              # Trained model files
├── splits/                             # Train/validation/test splits
├── step2_frozen_artifacts/             # Meta pipeline artifacts
└── frozen_baseline_splits/             # Baseline evaluation data
```

## Getting Started

### Prerequisites

```bash
pip install torch transformers scikit-learn pandas numpy
pip install streamlit plotly seaborn matplotlib
pip install google-genai  # For Gemini model
```

### Running the Models

1. **Baseline Models**:
   ```bash
   jupyter notebook base1_model_building.ipynb
   ```

2. **Novelty Model**:
   ```bash
   jupyter notebook novelty_model.ipynb
   ```

3. **Gemini LLM**:
   ```bash
   # Set your API key
   export GEMINI_API_KEY="your_api_key_here"
   jupyter notebook Gemini_model.ipynb
   ```

4. **Interactive Dashboard**:
   ```bash
   streamlit run humor_dashboard.py
   ```

## Model Performance

The project includes comprehensive evaluation across multiple metrics:

- **Macro F1-Score**: Balanced performance across all classes
- **Per-class Metrics**: Precision, recall, F1 for each humor category
- **Demographic Analysis**: Performance across different user groups
- **Calibration Metrics**: Uncertainty quantification quality

### Key Results
- Baseline DistilBERT: Establishes strong foundation
- Novelty RoBERTa: Improved demographic-aware predictions
- Gemini LLM: Competitive few-shot performance
- Meta Pipeline: Best overall performance through ensemble

## Interactive Features

### Streamlit Dashboard
- **Model Comparison**: Side-by-side evaluation of different approaches
- **Visualization**: Performance metrics and prediction distributions
- **Explainability**: Feature importance and attention visualization

##  Technical Highlights

### Advanced Techniques
- **LoRA Fine-tuning**: Efficient adaptation of large language models
- **Cost-sensitive Learning**: Addresses class imbalance in humor perception
- **Calibrated Classification**: Provides confidence scores with predictions
- **Cross-cultural Validation**: Ensures model generalization across demographics

### Evaluation Framework
- **Stratified Splits**: Maintains demographic balance across splits
- **Multiple Metrics**: Comprehensive evaluation beyond accuracy
- **Statistical Testing**: Significance testing for model comparisons
- **Error Analysis**: Deep dive into model failures and biases

## Future Work

- **Multi-modal Integration**: Incorporate visual humor (memes, comics)
- **Temporal Dynamics**: Account for changing humor preferences over time
- **Contextual Understanding**: Consider conversation context for humor
- **Cultural Adaptation**: Fine-tune for specific cultural groups
- **Real-time Learning**: Adapt to user feedback dynamically

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{personalised_humor_classification,
  title={Personalised Humor Classification using Large Language Models},
  author={Pratishtha Singh},
  year={2025},
  url={https://github.com/Pratishtha-Singh/Personalised-Humor-Classification-using-LLMs}
}
```



---

**Note**: This project is part of ongoing research in computational humor and personalized AI systems. The models and techniques demonstrated here contribute to understanding how AI can better understand human preferences and cultural contexts.Personalised-Humor-Classification-using-LLMs
Personalised humour classification with three labels: funny / not_funny / don’t_understand, using a cross-cultural dataset and user context (age, gender, ethnicity). Includes reproducible baselines (TF-IDF+LR, DistilBERT), a RoBERTa+demographics (LoRA) novelty model, and a two-stage meta pipeline with calibrated evaluation and explainability.
