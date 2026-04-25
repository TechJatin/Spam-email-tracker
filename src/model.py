from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
import joblib

class SpamDetector:
    def __init__(self, model_type='naive_bayes'):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = self._initialize_model()
        self.is_trained = False
    
    def _initialize_model(self):
        """Initialize the selected model"""
        if self.model_type == 'naive_bayes':
            return MultinomialNB()
        elif self.model_type == 'svm':
            return SVC(probability=True)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X, y):
        """Train the model"""
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)
        self.is_trained = True
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        predictions = self.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions),
            'recall': recall_score(y, predictions),
            'f1_score': f1_score(y, predictions)
        }
        return metrics
    
    def save(self, filepath):
        """Save model to file"""
        joblib.dump((self.model, self.vectorizer), filepath)
    
    def load(self, filepath):
        """Load model from file"""
        self.model, self.vectorizer = joblib.load(filepath)
        self.is_trained = True
