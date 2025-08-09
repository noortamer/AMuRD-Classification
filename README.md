# Product Category Prediction using Multilingual Embeddings

## Overview
This project predicts product categories based on their names using semantic embeddings and cosine similarity.  
It leverages the `intfloat/multilingual-e5-base` model to produce multilingual sentence embeddings for accurate category matching across diverse product datasets.

The pipeline:
1. Cleans and preprocesses the dataset.
2. Generates embeddings for product names and reference category names.
3. Predicts categories using cosine similarity.
4. Evaluates performance against a baseline (Dummy Classifier).
5. Handles category unification to fix semantic overlaps like:
   - "Sweets & Desserts" → "Chocolates, Sweets & Desserts"

---

## Model Details

I use intfloat/multilingual-e5-base:

	•	A sentence transformer model trained for multilingual retrieval and semantic search.
	•	Supports over 100 languages.
	•	Outputs 768-dimensional embeddings.

Embedding process:

	1.	Tokenize text with the model’s tokenizer.
	2.	Pass through the transformer model.
	3.	Extract the [CLS] token representation as the embedding vector.
	4.	Compare product embeddings to category embeddings using cosine similarity.

 ---

 ## Evaluation Results

### Category Prediction Performance
After unifying overlapping categories:

| Metric       | Value |
|--------------|-------|
| Accuracy     | 0.27  |
| Macro Avg F1 | 0.13  |
| Weighted F1  | 0.29  |

### Dummy Model Performance
A baseline model that always predicts the most common class:

| Metric  | Value     |
|---------|-----------|
| F1 Score | 0.0239   |

### Improvement Over Dummy Model
the embedding-based model achieves an **over 12× improvement** in weighted F1 score compared to the dummy baseline.

---

## Possible Improvements

- Fine-tuning the embedding model on domain-specific product data.
- Data balancing to handle underrepresented categories.
- Multi-label classification for products belonging to multiple categories.
- Context-aware embeddings (adding brand, description, packaging info).
