  ²# Sentiment Analysis API - MLOps Pipeline

![Tests](https://github.com/jeoram/Sentiment_Analysis_PartII/actions/workflows/test.yml/badge.svg)
![Evaluation](https://github.com/jeoram/Sentiment_Analysis_PartII/actions/workflows/evaluate.yml/badge.svg)

A complete MLOps pipeline for sentiment analysis with Docker containerization and GitHub Actions CI/CD.

## 🚀 Features

- **FastAPI REST API** for sentiment prediction
- **Docker containerization** with multi-service architecture
- **GitHub Actions CI/CD** pipeline with automated testing and deployment
- **Model evaluation** with performance thresholds

## 📁 Project Structure

```
MLOps_Part II/
├── .github/workflows/      # GitHub Actions workflows
│   ├── test.yml           # Tests & linting
│   ├── evaluate.yml       # Model evaluation
│   └── build.yml          # Docker build & push
├── src/                   # Source code
│   ├── api.py            # FastAPI application
│   ├── model.py          # Sentiment model
│   └── evaluate.py       # Model evaluation script
├── tests/                 # Unit tests
│   └── test_api.py       # API tests
├── models/               # Trained models (volume)
├── data/                 # Datasets (volume)
├── logs/                 # Application logs (volume)
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Multi-service setup
└── requirements.txt      # Python dependencies
```

## 🐳 Docker Usage

### Build and run with Docker Compose
```bash
# Start all services
docker-compose up -d

# Start with MongoDB UI (development)
docker-compose --profile dev up -d

# View logs
docker-compose logs -f api
```

### Build manually
```bash
docker build -t sentiment-analysis-api .
docker run -p 8000:8000 sentiment-analysis-api
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/health` | GET | Health check |
| `/model/info` | GET | Model information |
| `/predict` | POST | Predict sentiment |
| `/predict/batch` | POST | Batch prediction |
| `/docs` | GET | Swagger UI |

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

### Example Response
```json
{
  "text": "I love this product!",
  "sentiment": "positive",
  "confidence": 0.85,
  "timestamp": "2024-01-15T10:30:00"
}
```

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src
```

## 🔄 CI/CD Pipeline

The GitHub Actions pipeline consists of 3 workflows:

1. **test.yml** - Runs on every push/PR
   - Code linting (flake8, black, isort)
   - Unit tests with coverage

2. **evaluate.yml** - Runs after tests pass
   - Evaluates model performance
   - Fails if metrics below threshold

3. **build.yml** - Runs after evaluation passes
   - Builds Docker image
   - Pushes to Docker Hub

## 📊 Model Metrics Thresholds

| Metric | Minimum Threshold |
|--------|------------------|
| Accuracy | 0.80 |
| F1 Score | 0.75 |
| Precision | 0.75 |
| Recall | 0.70 |

## ⚙️ GitHub Secrets Required

| Secret | Description |
|--------|-------------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_PASSWORD` | Your Docker Hub password |

## 👥 Authors & Partener

- [Edward & Kennedy]

## 📄 License

MIT License
