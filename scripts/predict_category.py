# scripts/predict_category.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.dataset import load_and_clean_dataset
from modules.model import load_embedding_model, encode_texts, compute_cosine_similarity


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

# === Load Dataset ===
df = load_and_clean_dataset()

# Filter classes with more than 1 sample
class_counts = df['category'].value_counts()
valid_classes = class_counts[class_counts > 1].index
df = df[df['category'].isin(valid_classes)].reset_index(drop=True)

# === Encode Categories ===
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['category'])

# === Load Model ===
model = SentenceTransformer("intfloat/multilingual-e5-base")

# === Embed Product Names and Categories ===
product_embeddings = model.encode(
    ["query: " + name for name in tqdm(df['embedding_text'])], show_progress_bar=True
)
category_embeddings = model.encode(
    ["passage: " + label for label in label_encoder.classes_], show_progress_bar=True
)

# === Predict Closest Category ===
predicted_indices = []

for emb in product_embeddings:
    similarities = cosine_similarity([emb], category_embeddings)
    pred_index = np.argmax(similarities)
    predicted_indices.append(pred_index)

df['predicted_label'] = predicted_indices
df['predicted_category'] = label_encoder.inverse_transform(predicted_indices)

# === Evaluation ===
print("\n Category Prediction Report:")
print(classification_report(df['label'], df['predicted_label'], target_names=label_encoder.classes_))

# === Dummy Classifier for Baseline Comparison ===
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(np.zeros((len(df), 1)), df['label'])
dummy_preds = dummy.predict(np.zeros((len(df), 1)))
print("\n🧪 Dummy Model F1 Score:", f1_score(df['label'], dummy_preds, average='weighted'))

# Save embeddings to .npy files
np.save("outputs/product_embeddings.npy", product_embeddings)
np.save("outputs/category_embeddings.npy", category_embeddings)

# Save the DataFrame with predictions
df.to_csv("outputs/predicted_categories.csv", index=False)
