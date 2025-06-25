# Chat-Moderation Repository

A comprehensive suite of notebooks and scripts for English chat moderation tasks: data cleaning, dataset balancing, training, evaluation, testing, and deployment of moderation models.

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [File Structure](#file-structure)
4. [Installation](#installation)
5. [Usage Guide](#usage-guide)
   - [1. Data Cleaning](#1-data-cleaning)
   - [2. Model Training](#2-model-training)
   - [3. Accuracy Analysis](#3-accuracy-analysis)
   - [4. Final Testing](#4-final-testing)
   - [5. Using Pretrained Models](#5-using-pretrained-models)
   - [6. Multilingual Support](#6-multilingual-support)
6. [Specifications](#specifications)
7. [Dependencies](#dependencies)

---

## Overview

This repository provides end-to-end notebooks for building, evaluating, testing, and deploying an English-language chat moderation pipeline. From raw data cleaning to balanced dataset creation, model training, evaluation, and final testing, it serves as a reference and toolkit for researchers and engineers working on automated content moderation.

## Features

- **Data Cleaning:** Remove noise, normalize text, strip unwanted characters.
- **Dataset Balancing:** Techniques for ensuring label distribution parity across labels and decisions.
- **Model Training:** Training classification models for detecting harmful or off-topic messages.
- **Evaluation & Accuracy Analysis:** Compute accuracy, precision, recall, F1 scores.
- **Testing:** Validate trained models on held-out test data and inspect predictions.
- **Model Artifacts:** Persist trained models, encoders, and tokenizers for reuse.
- **Pretrained Model Usage:** Load saved models to predict on new data instantly.
- **Multilingual Support:** Run full pipeline for non-English languages including German, Spanish, French, Portuguese, and Russian.

## File Structure

```
├── Clean_English.ipynb           # Initial cleaning of raw chat data
├── Clean_Balanced_English.ipynb  # Cleaning + balancing of label and decision fields
├── Train_English.ipynb           # Model training notebook (formerly Test_English)
├── Accurarcy_English.ipynb       # Evaluation metrics & accuracy analysis
├── CHAT_English.ipynb            # Testing and inference using trained models
├── save/                         # Serialized artifacts (tokenizer, models, encoders, etc.)
│   ├── label_encoder.pkl
│   ├── model_state.pt
│   ├── special_tokens_map
│   ├── status_encoder.pkl
│   ├── tokenizer/
│   ├── tokenizer_config
│   └── vocab
├── Chat_Languages/                 # Chat moderation pipelines for other languages
    ├── Clean_Languages.ipynb
    ├── Train_Languages.ipynb
    ├── Accurarcy_Languages.ipynb
    ├── CHAT_Languages.ipynb
    └── save/                     # Language-specific model artifacts
```

### Multilingual Language Codes Supported

- `de`: German
- `es`: Spanish
- `fr`: French
- `pt`: Portuguese
- `ru`: Russian

Each notebook in the `multilingual/` folder follows the same structure and logic as the English pipeline:

- Clean the dataset using `Clean_Languages.ipynb`
- Train a moderation model with `Train_Languages.ipynb`
- Evaluate it using `Accurarcy_Languages.ipynb`
- Test predictions using `CHAT_Languages.ipynb`

Language can be selected dynamically within the notebooks via language code.

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/chat-moderation.git
   cd chat-moderation
   ```

2. **Create a Python virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** Ensure `requirements.txt` includes packages like `pandas`, `scikit-learn`, `imblearn`, `nltk`, `matplotlib`, `torch`, and `transformers` (if applicable).

## Usage Guide

### 1. Data Cleaning

Use this stage to prepare your dataset before model training. You can choose between basic cleaning or apply balancing strategies to handle class imbalance.

- Open either `Clean_English.ipynb` or `Clean_Balanced_English.ipynb` depending on your needs.
- Customize preprocessing and balancing methods inside the notebook.
- Output a processed CSV file that fits your training needs (e.g., `cleaned_data.csv` or `balanced_data.csv`).

### 2. Model Training

1. Open `Train_English.ipynb`.
2. Load cleaned or balanced CSV file.
3. Define and train the model.
4. Save artifacts to `save/` folder for future use.

### 3. Accuracy Analysis

1. Launch `Accurarcy_English.ipynb`.
2. Use saved model artifacts and training data.
3. Evaluate performance and visualize metrics.

### 4. Final Testing

1. Open `CHAT_English.ipynb`.
2. Load artifacts from the `save/` folder.
3. Run inference on new messages and display predicted labels/decisions.

### 5. Using Pretrained Models

You can use the pretrained model directly without retraining:

1. Ensure the following files are present in the `save/` folder:

   - `model_state.pt`
   - `label_encoder.pkl`
   - `status_encoder.pkl`
   - `tokenizer/` directory and associated configs

2. Open `CHAT_English.ipynb`

3. Provide your own input text or a CSV file of messages.

4. Run the notebook to generate predictions using the pretrained model.

This allows fast deployment and testing without repeating the training pipeline.

### 6. Multilingual Support

For moderation in other languages, navigate to the `Chat_Languages/` folder and repeat the steps outlined above for English:

1. Clean the dataset using `Clean_Languages.ipynb`
2. Train the model with `Train_Languages.ipynb`
3. Evaluate with `Accurarcy_Languages.ipynb`
4. Test predictions using `CHAT_Languages.ipynb`

> You can specify the language code (`de`, `es`, `fr`, `pt`, `ru`) inside each notebook to adapt the tokenizer and pipeline.

## Specifications

| Component            | Details                                                                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Input Format         | CSV files with columns: `id`, `text`, `label`, `decision`                                                                             |
| Cleaning Steps       | Lowercase, regex filtering, stopwords, tokenization                                                                                   |
| Balancing Techniques | Class balancing across `label` and `decision` (undersampling/SMOTE)                                                                   |
| Feature Extraction   | TF-IDF vectors, word embeddings, HuggingFace transformers                                                                             |
| Models Supported     | Logistic Regression, Random Forest, LSTM, Transformers                                                                                |
| Evaluation Metrics   | Accuracy, Precision, Recall, F1-score, Confusion Matrix                                                                               |
| Output Files         | `cleaned_data.csv`, `balanced_data.csv`, `model_state.pt`, `predictions.csv`, `label_encoder.pkl`, `status_encoder.pkl`, `tokenizer/` |

## Dependencies

Make sure the following (or higher) versions are installed:

- Python 3.8+
- pandas
- numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- nltk
- matplotlib
- torch
- transformers

```text
# Example requirements.txt contents
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
imbalanced-learn>=0.9
nltk>=3.6
matplotlib>=3.4
torch>=1.9
transformers>=4.10
```


---

