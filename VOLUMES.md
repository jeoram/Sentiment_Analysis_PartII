# Docker Volumes Configuration

## Overview

This project uses Docker volumes to persist important data across container restarts and rebuilds.

## Volumes Structure

```
├── models/          # Trained ML models (VOLUME)
│   ├── sentiment_model.joblib
│   └── vectorizer.joblib
│
├── data/            # Datasets (VOLUME)
│   ├── train.csv
│   ├── test.csv
│   └── predictions.csv
│
└── logs/            # Application logs (VOLUME)
    ├── app.log
    ├── predictions.log
    └── metrics.log
```

## Why Use Volumes?

| Volume | Purpose | Risk without volume |
|--------|---------|---------------------|
| `models/` | Store trained models | Models lost on container restart, need to retrain |
| `data/` | Store datasets | Data lost, can't reproduce experiments |
| `logs/` | Store logs & metrics | Lose prediction history and audit trail |

## Volume Commands

### Create named volumes
```bash
docker volume create sentiment-models
docker volume create sentiment-data
docker volume create sentiment-logs
```

### Run with volumes
```bash
docker run -d \
  -v sentiment-models:/app/models \
  -v sentiment-data:/app/data \
  -v sentiment-logs:/app/logs \
  -p 8000:8000 \
  sentiment-analysis-api
```

### Or use bind mounts (for development)
```bash
docker run -d \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -p 8000:8000 \
  sentiment-analysis-api
```

## Docker Compose (Recommended)

The `docker-compose.yml` file automatically configures these volumes. See the compose file for details.
