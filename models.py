import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import joblib

def train_classifier(csv_path: str, model_path: str = "saved_model/email_classifier.pkl"):
    df = pd.read_csv(csv_path)
    if 'email' not in df.columns or 'type' not in df.columns:
        raise ValueError("CSV must have 'email' and 'type' columns.")

    X = df['email']
    y = df['type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    clf_pipeline.fit(X_train, y_train)

    preds = clf_pipeline.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, preds))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf_pipeline, model_path)
    print(f"Model saved to {model_path}")