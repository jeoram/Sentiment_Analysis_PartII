"""
Unit Tests for Sentiment Analysis API
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app, model
from src.model import SentimentModel


# Ensure model is loaded before tests
@pytest.fixture(scope="module", autouse=True)
def setup_model():
    """Load model before running tests"""
    if not model.is_loaded():
        model.load()
    yield


# Test client
client = TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint"""

    def test_health_check_returns_200(self):
        """Health endpoint should return 200"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_returns_healthy_status(self):
        """Health endpoint should return healthy status"""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestRootEndpoint:
    """Tests for / endpoint"""

    def test_root_returns_200(self):
        """Root endpoint should return 200"""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_welcome_message(self):
        """Root endpoint should return welcome message"""
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert "Welcome" in data["message"]


class TestModelInfoEndpoint:
    """Tests for /model/info endpoint"""

    def test_model_info_returns_200(self):
        """Model info endpoint should return 200"""
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_contains_required_fields(self):
        """Model info should contain required fields"""
        response = client.get("/model/info")
        data = response.json()
        assert "model_name" in data
        assert "version" in data
        assert "classes" in data


class TestPredictEndpoint:
    """Tests for /predict endpoint"""

    def test_predict_positive_sentiment(self):
        """Should predict positive sentiment for positive text"""
        response = client.post("/predict", json={"text": "I love this! It's amazing and wonderful!"})
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] in ["positive", "negative", "neutral"]
        assert data["confidence"] > 0

    def test_predict_negative_sentiment(self):
        """Should predict negative sentiment for negative text"""
        response = client.post("/predict", json={"text": "I hate this! It's terrible and awful!"})
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] in ["positive", "negative", "neutral"]
        assert data["confidence"] > 0

    def test_predict_returns_required_fields(self):
        """Prediction should return all required fields"""
        response = client.post("/predict", json={"text": "This is a test text for sentiment analysis"})
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "sentiment" in data
        assert "confidence" in data
        assert "timestamp" in data

    def test_predict_empty_text_returns_error(self):
        """Empty text should return validation error"""
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_predict_missing_text_returns_error(self):
        """Missing text field should return validation error"""
        response = client.post("/predict", json={})
        assert response.status_code == 422





class TestIntegration:
    """Integration tests"""

    def test_full_prediction_flow(self):
        """Test complete prediction flow"""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200

        # 2. Get model info
        info_response = client.get("/model/info")
        assert info_response.status_code == 200

        # 3. Make prediction
        predict_response = client.post("/predict", json={"text": "This product is great!"})
        assert predict_response.status_code == 200
        data = predict_response.json()
        assert data["sentiment"] in ["positive", "negative", "neutral"]

    def test_multiple_predictions(self):
        """Test multiple sequential predictions"""
        texts = ["I love this!", "This is terrible", "It's okay I guess"]

        for text in texts:
            response = client.post("/predict", json={"text": text})
            assert response.status_code == 200
            data = response.json()
            assert "sentiment" in data
            assert "confidence" in data
