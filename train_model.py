"""
Email Spam Detection Model Training Script
Trains and saves the model and vectorizer for the web application
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
print("Loading dataset...")
df = pd.read_csv("spam.csv", encoding="latin-1")

# Data preprocessing
print("Preprocessing data...")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Display dataset info
print(f"\nDataset Shape: {df.shape}")
print(f"Label Distribution:\n{df['label'].value_counts()}")
print(f"Class Distribution:")
print(f"  - HAM (NOT SPAM): {(df['label']==0).sum()} messages ({(df['label']==0).sum()/len(df)*100:.2f}%)")
print(f"  - SPAM: {(df['label']==1).sum()} messages ({(df['label']==1).sum()/len(df)*100:.2f}%)")

# Split dataset
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Initialize TF-IDF Vectorizer
print("\nTraining TF-IDF Vectorizer...")
tfidf = TfidfVectorizer(max_features=3000, stop_words='english', lowercase=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"TF-IDF Features: {X_train_tfidf.shape[1]}")

# Train Multinomial Naive Bayes model
print("\nTraining Multinomial Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate model
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

y_pred_train = model.predict(X_train_tfidf)
y_pred_test = model.predict(X_test_tfidf)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=['HAM (NOT SPAM)', 'SPAM']))

print("\nConfusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_pred_test))

# Save model and vectorizer
print("\n" + "="*50)
print("SAVING MODEL AND VECTORIZER")
print("="*50)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved as 'model.pkl'")

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("✓ Vectorizer saved as 'vectorizer.pkl'")

print("\n✓ Training complete! Files ready for deployment.")
print("\nModel Performance Summary:")
print(f"  • Algorithm: Multinomial Naive Bayes")
print(f"  • Vectorizer: TF-IDF (max_features=3000)")
print(f"  • Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  • Training Accuracy: {train_accuracy*100:.2f}%")
