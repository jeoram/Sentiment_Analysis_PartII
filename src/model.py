"""
Sentiment Analysis Model
Trained on app reviews dataset with score-based sentiment labels
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Model paths
MODEL_DIR = os.environ.get("MODEL_PATH", "models")
DATA_DIR = os.environ.get("DATA_PATH", "data")
MODEL_FILE = os.path.join(MODEL_DIR, "sentiment_model.joblib")
DATASET_FILE = os.path.join(DATA_DIR, "dataset.csv")


def score_to_sentiment(score: int) -> str:
    """
    Convert numeric score (1-5) to sentiment label
    
    Args:
        score: Rating from 1 to 5
        
    Returns:
        Sentiment label: positive, negative, or neutral
    """
    if score <= 2:
        return "negative"
    elif score >= 4:
        return "positive"
    else:
        return "neutral"


class SentimentModel:
    """
    Sentiment Analysis Model
    
    Trained on app reviews dataset using TF-IDF and Logistic Regression.
    Score mapping:
        - 1-2 stars: negative
        - 3 stars: neutral
        - 4-5 stars: positive
    """
    
    def __init__(self):
        self.model = None
        self._loaded = False
        self.classes = ["negative", "neutral", "positive"]
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded
    
    def load(self) -> None:
        """Load the model from disk or train a new one"""
        try:
            if os.path.exists(MODEL_FILE):
                self.model = joblib.load(MODEL_FILE)
                logger.info(f"Model loaded from {MODEL_FILE}")
                self._loaded = True
            else:
                logger.info("No saved model found, training new model...")
                self.train_from_dataset()
                self._loaded = True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback: train a new model
            self.train_from_dataset()
            self._loaded = True
    
    def load_dataset(self) -> Tuple[list, list]:
        """
        Load and preprocess the reviews dataset
        
        Returns:
            Tuple of (texts, labels)
        """
        if not os.path.exists(DATASET_FILE):
            raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")
        
        logger.info(f"Loading dataset from {DATASET_FILE}")
        df = pd.read_csv(DATASET_FILE)
        
        # Clean data
        df = df.dropna(subset=['content', 'score'])
        df['content'] = df['content'].astype(str)
        df['score'] = df['score'].astype(int)
        
        # Convert scores to sentiment labels
        df['sentiment'] = df['score'].apply(score_to_sentiment)
        
        logger.info(f"Dataset loaded: {len(df)} samples")
        logger.info(f"Distribution: {df['sentiment'].value_counts().to_dict()}")
        
        return df['content'].tolist(), df['sentiment'].tolist()
    
    def train_from_dataset(self) -> Dict[str, float]:
        """
        Train the model using the reviews dataset
        
        Returns:
            Training metrics
        """
        try:
            texts, labels = self.load_dataset()
        except FileNotFoundError:
            logger.warning("Dataset not found, using sample data")
            texts, labels = self._get_sample_data()
        
        return self.train(texts, labels)
    
    def _get_sample_data(self) -> Tuple[list, list]:
        """Get sample training data as fallback"""
        sample_data = [
            ("I love this app! It's amazing!", "positive"),
            ("Best app ever, highly recommend!", "positive"),
            ("Great features and easy to use", "positive"),
            ("Wonderful experience, 5 stars!", "positive"),
            ("This app is terrible", "negative"),
            ("Waste of money, doesn't work", "negative"),
            ("I hate this app, full of bugs", "negative"),
            ("Worst app I've ever used", "negative"),
            ("It's okay, nothing special", "neutral"),
            ("Average app, does the job", "neutral"),
            ("Not bad but not great either", "neutral"),
            ("It's fine I guess", "neutral"),
        ]
        return [t[0] for t in sample_data], [t[1] for t in sample_data]
    
    def train(self, texts: list, labels: list) -> Dict[str, float]:
        """
        Train the model on labeled data
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels
            
        Returns:
            Training metrics
        """
        logger.info(f"Training model on {len(texts)} samples...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )),
            ('clf', LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ))
        ])
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Training complete! Accuracy: {accuracy:.4f}")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Save model
        self.save()
        
        return {
            "train_accuracy": self.model.score(X_train, y_train),
            "test_accuracy": accuracy,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment of text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with sentiment and confidence
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Get prediction and probabilities
        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        confidence = float(max(probabilities))
        
        # Get probability for each class (convert numpy types to native Python)
        class_probs = {
            str(cls): float(prob) 
            for cls, prob in zip(self.model.classes_, probabilities)
        }
        
        return {
            "sentiment": str(prediction),  # Convert numpy string to Python string
            "confidence": round(confidence, 4),
            "probabilities": class_probs
        }
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        predictions = self.model.predict(texts)
        probabilities = self.model.predict_proba(texts)
        
        results = []
        for i, text in enumerate(texts):
            results.append({
                "text": text,
                "sentiment": predictions[i],
                "confidence": round(float(max(probabilities[i])), 4)
            })
        
        return results
    
    def save(self) -> None:
        """Save the model to disk"""
        if self.model is not None:
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(self.model, MODEL_FILE)
            logger.info(f"Model saved to {MODEL_FILE}")
        else:
            logger.warning("No model to save")
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": "SentimentClassifier",
            "version": "1.0.0",
            "classes": self.classes,
            "description": "App reviews sentiment classifier using TF-IDF + Logistic Regression",
            "score_mapping": {
                "1-2 stars": "negative",
                "3 stars": "neutral",
                "4-5 stars": "positive"
            }
        }
