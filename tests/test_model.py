"""
Unit Tests for Sentiment Model
"""

import os
import pytest
from unittest.mock import MagicMock, patch, mock_open
import joblib
import pandas as pd
import numpy as np
from src.model import SentimentModel, score_to_sentiment, MODEL_FILE

# Test score_to_sentiment function
def test_score_to_sentiment():
    assert score_to_sentiment(1) == "negative"
    assert score_to_sentiment(2) == "negative"
    assert score_to_sentiment(3) == "neutral"
    assert score_to_sentiment(4) == "positive"
    assert score_to_sentiment(5) == "positive"
    assert score_to_sentiment(0) == "negative" # Should handle boundary
    assert score_to_sentiment(6) == "positive" # Should handle boundary

class TestSentimentModel:
    
    def setup_method(self):
        self.model = SentimentModel()

    def test_initialization(self):
        assert self.model.model is None
        assert self.model._loaded is False
        assert self.model.classes == ["negative", "neutral", "positive"]

    def test_load_existing_model(self):
        # Mock os.path.exists to return True
        # Mock joblib.load to return a dummy model
        with patch("os.path.exists", return_value=True), \
             patch("joblib.load", return_value="dummy_model"):
            
            self.model.load()
            
            assert self.model.model == "dummy_model"
            assert self.model.is_loaded() is True

    def test_load_no_model_trains_new(self):
        # Mock os.path.exists to return False (no model file)
        # Mock train_from_dataset
        with patch("os.path.exists", return_value=False), \
             patch.object(self.model, "train_from_dataset") as mock_train:
            
            self.model.load()
            
            mock_train.assert_called_once()
            assert self.model.is_loaded() is True

    def test_predict_without_loading_raises_error(self):
        with pytest.raises(ValueError, match="Model not loaded"):
            self.model.predict("test text")

    def test_predict_returns_correct_structure(self):
        # Mock the internal sklearn logic
        mock_sklearn_model = MagicMock()
        mock_sklearn_model.predict.return_value = np.array(["positive"])
        mock_sklearn_model.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])
        mock_sklearn_model.classes_ = ["negative", "neutral", "positive"]
        
        self.model.model = mock_sklearn_model
        # Manually set loaded to True for testing predict directly
        
        result = self.model.predict("This is great")
        
        assert result["sentiment"] == "positive"
        assert result["confidence"] == 0.8
        assert "positive" in result["probabilities"]
        assert result["probabilities"]["positive"] == 0.8

    def test_train_from_dataset_handles_missing_file(self):
        # Should fallback to sample data if file not found
        with patch.object(self.model, "load_dataset", side_effect=FileNotFoundError), \
             patch.object(self.model, "train") as mock_train:
             
            self.model.train_from_dataset()
            
            # Should call train with *some* data (sample data)
            mock_train.assert_called_once()

    def test_save_model(self):
        self.model.model = "dummy_model"
        
        with patch("os.makedirs") as mock_makedirs, \
             patch("joblib.dump") as mock_dump:
             
            self.model.save()
            
            mock_makedirs.assert_called()
            mock_dump.assert_called_with("dummy_model", MODEL_FILE)

    def test_get_info(self):
        info = self.model.get_info()
        assert info["model_name"] == "SentimentClassifier"
        assert "version" in info
        assert "classes" in info
