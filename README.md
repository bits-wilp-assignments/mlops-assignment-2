# Pet Adoption Classification - MLOps Project

A production-ready MLOps pipeline for cat vs dog image classification using TensorFlow, MLflow, DVC, and Kubernetes.

## Project Structure

```
mlops-assignment-2/
├── common/                 # Shared configuration and utilities
│   ├── base.py            # Common constants and MLflow config
│   └── logger.py          # Logging utilities
├── training/              # Model training pipeline
│   ├── src/
│   │   ├── data/         # Data preprocessing
│   │   ├── model/        # Training, evaluation, validation
│   │   └── config/       # Training configuration
│   └── test/             # Training unit tests
├── serving/               # API serving
│   ├── src/
│   │   ├── api/          # FastAPI endpoints
│   │   ├── inference/    # Model prediction logic
│   │   └── config/       # Serving configuration
│   ├── monitoring/       # Prometheus metrics
│   └── test/             # Serving unit tests
├── deployment/
│   └── k8s/              # Kubernetes manifests
├── data/
│   ├── raw/              # Raw dataset (DVC tracked)
│   └── processed/        # Preprocessed images
├── models/               # Trained models (MLflow tracked)
├── .github/workflows/    # CI/CD pipelines
├── dvc.yaml              # DVC pipeline definition
└── smoketest.py          # End-to-end smoke tests
```

## Setup Instructions

### Prerequisites
- Python 3.12.0
- Docker (for containerized serving)
- Kubernetes cluster (for production deployment)
- MLflow tracking server running at `http://localhost:5050`

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd mlops-assignment-2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For training module only
pip install -r training/requirements.txt

# For serving module only
pip install -r serving/requirements.txt
```

### Data Setup

#### DVC with Google Drive Integration

This project uses DVC (Data Version Control) to manage large datasets stored on Google Drive.

**Initial Setup:**

```bash
# Install DVC with Google Drive support
pip install 'dvc[gdrive]'

# Initialize DVC (if not already initialized)
dvc init

# Add Google Drive as remote storage
dvc remote add -d myremote gdrive://1ByDvwOz1ZFaFXaXj-ax_OEcREfZS4Ewi

# Configure DVC to use service account (for CI/CD)
dvc remote modify myremote gdrive_use_service_account true
dvc remote modify myremote gdrive_service_account_json_file_path .dvc/tmp/gdrive-sa.json
```

**Authentication Options:**

1. **Interactive OAuth (Local Development):**
   ```bash
   # First time: DVC will prompt you to authenticate via browser
   dvc pull data/raw.dvc
   # Follow the OAuth flow to grant access to Google Drive
   ```

2. **Service Account (CI/CD):**
   ```bash
   # Create a service account in Google Cloud Console
   # Download the JSON key file
   # Store it as a GitHub secret: GDRIVE_CREDENTIALS_DATA
   
   # In CI/CD, the credentials are automatically configured:
   mkdir -p .dvc/tmp
   echo "$GDRIVE_CREDENTIALS_DATA" > .dvc/tmp/gdrive-sa.json
   ```

**Working with Data:**

```bash
# Pull raw data from Google Drive (25K images, ~850MB)
dvc pull data/raw.dvc

# Check data status
dvc status

# If you modify the raw data, track changes:
dvc add data/raw
dvc push data/raw.dvc
git add data/raw.dvc
git commit -m "Update raw dataset"
```

**Data Structure:**
```
data/raw/PetImages/
├── Cat/          # ~12,500 cat images
│   ├── cat.0.jpg
│   ├── cat.1.jpg
│   └── ...
└── Dog/          # ~12,500 dog images
    ├── dog.0.jpg
    ├── dog.1.jpg
    └── ...
```

**Manual Setup (Alternative):**

If you prefer not to use DVC:
```bash
# Download the Kaggle dataset: Dogs vs. Cats
# https://www.microsoft.com/en-us/download/details.aspx?id=54765

# Extract and place in data/raw/PetImages/
# Ensure structure: data/raw/PetImages/Cat/ and data/raw/PetImages/Dog/
```

### Automated Pipelines

This project includes three GitHub Actions workflows that automate the entire MLOps lifecycle:

1. **CI Pipeline** (`.github/workflows/ci.yml`): Automatically runs serving tests and builds/pushes Docker images whenever code changes are pushed to serving or common modules.
2. **Training Pipeline** (`.github/workflows/training.yml`): Executes the complete DVC training pipeline (preprocess → train → validate) on schedule or when training code changes, automatically promoting champion models.
3. **CD Pipeline** (`.github/workflows/cd.yml`): Deploys the latest Docker image to Kubernetes after successful CI builds, ensuring zero-downtime rolling updates.

## Running Training

### Using DVC Pipeline (Recommended)

```bash
# Run the complete pipeline: preprocess → train → validate
dvc repro

# Run individual stages
dvc repro preprocess  # Preprocess raw images
dvc repro train       # Train model and log to MLflow
dvc repro validate    # Validate and promote model to 'champion'
```

### Manual Training

```bash
# Step 1: Preprocess data
python -m training.src.data.preprocess

# Step 2: Train model
python -m training.src.model.train

# Step 3: Validate model (replace <run-id> with MLflow run ID)
python -m training.src.model.validate --run-id <run-id>
```

### Training Tests

```bash
# Run all training tests
pytest training/test/ -v

# Run specific test files
pytest training/test/test_train.py -v
pytest training/test/test_preprocess.py -v
pytest training/test/test_validate.py -v
```

### Training Configuration

Edit `training/src/config/settings.py`:
```python
BATCH_SIZE = 32
EPOCHS = 5
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'binary_crossentropy'
USE_MIXED_PRECISION = False  # Enable for GPU acceleration
VALIDATION_METRIC_NAME = 'test_accuracy'
```

## Running CI/CD

### Continuous Integration (CI)

**Trigger**: Push to `main` or pull requests affecting `serving/**`, `common/**`

**Workflow**: `.github/workflows/ci.yml`

**Jobs**:
1. Run serving tests with pytest
2. Build Docker image
3. Push to Docker Hub (on main branch only)

```bash
# CI runs automatically on push/PR
# To test locally:
pytest serving/test/ -v
docker build -f serving/Dockerfile -t pet-adoptation-api .
```

### Training Pipeline CI

**Trigger**: Push to `main` affecting `training/**`, `common/**`, or weekly schedule

**Workflow**: `.github/workflows/training.yml`

**Jobs**:
1. Run training tests
2. Pull data with DVC
3. Execute DVC pipeline (preprocess → train → validate)

```bash
# Training CI runs automatically
# To test locally:
pytest training/test/ -v
dvc repro
```

### Continuous Deployment (CD)

**Trigger**: After successful CI workflow or manual dispatch

**Workflow**: `.github/workflows/cd.yml`

**Jobs**:
1. Deploy to Kubernetes cluster
2. Apply all manifests (namespace, deployment, service, HPA, PDB)
3. Wait for rollout completion

```bash
# CD runs automatically after CI
# To deploy manually:
kubectl apply -f deployment/k8s/
```

## Running Deployment

### Local Development

```bash
# Start the FastAPI server
python -m serving.src.api.main

# Or using uvicorn with auto-reload
uvicorn serving.src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Deployment

```bash
# Build Docker image
docker build -f serving/Dockerfile -t pet-adoptation-api .

# Run container
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5050 \
  pet-adoptation-api

# Test the container
curl http://localhost:8000/health
```

### Kubernetes Deployment

```bash
# Apply all Kubernetes manifests
kubectl apply -f deployment/k8s/

# Check deployment status
kubectl get pods -n pet-adoptation
kubectl get svc -n pet-adoptation

# Check rollout status
kubectl rollout status deployment/pet-adoptation-api -n pet-adoptation

# View logs
kubectl logs -f deployment/pet-adoptation-api -n pet-adoptation

# Access the service (NodePort 31000)
curl http://<node-ip>:31000/health
```

## API Details

### Health Endpoint

**GET** `/health`

Check API health and model information.

```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "ok",
  "model_name": "pet-classification-model",
  "model_alias": "champion"
}
```

### Predict Endpoint

**PUT** `/predict`

Classify an uploaded pet image as Cat or Dog.

```bash
curl -X PUT http://localhost:8000/predict \
  -F "file=@data/raw/PetImages/Cat/cat.0.jpg"
```

**Response**:
```json
{
  "label": "Cat",
  "probability": 0.987
}
```

### Metrics Endpoint

**GET** `/metrics`

Prometheus-formatted metrics for monitoring.

```bash
curl http://localhost:8000/metrics
```

**Metrics exposed**:
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request duration histogram

## API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to:
- Explore all available endpoints
- Test API calls directly from the browser
- View request/response schemas
- Download OpenAPI specification

## Smoke Tests

### Serving API Smoke Tests

The smoke tests provide quick end-to-end validation of the deployed serving API to ensure it's functioning correctly in production or staging environments.

**Prerequisites:**
- API server must be running on `http://localhost:8000`
- Sample test image must exist at `data/raw/PetImages/Cat/cat.0.jpg`

**Running Smoke Tests:**

```bash
# Start the API server first (in a separate terminal)
python -m serving.src.api.main

# Run smoke tests
python smoketest.py
```

**Test Coverage:**

1. **Health Check Test** (`test_health`)
   - **Purpose**: Verifies API is running and responsive
   - **Endpoint**: `GET /health`
   - **Validation**: 
     - Status code is 200
     - Response contains model information
   - **Success Criteria**: API returns healthy status

2. **Prediction Test** (`test_predict`)
   - **Purpose**: Validates end-to-end image classification pipeline
   - **Endpoint**: `POST /predict`
   - **Test Data**: Cat image from dataset (`data/raw/PetImages/Cat/cat.0.jpg`)
   - **Validation**:
     - Status code is 200
     - Response contains `label` and `probability` fields
     - Label is either "Cat" or "Dog"
     - Probability is between 0 and 1
   - **Success Criteria**: Image is correctly classified

**Expected Output:**
```
Health OK
Prediction OK {'label': 'Cat', 'probability': 0.987}
```

**Testing Against Different Environments:**

```python
# Production environment
import os
os.environ['API_URL'] = 'http://your-production-url.com'
python smoketest.py

# Kubernetes service
kubectl port-forward svc/pet-adoptation-api 8000:80 -n pet-adoptation
python smoketest.py

# Docker container
docker run -p 8000:8000 pet-adoptation-api
python smoketest.py
```

**Smoke Test Source Code:**

```python
import requests

def test_health():
    """Verify API health endpoint returns 200 and model info"""
    resp = requests.get("http://localhost:8000/health")
    assert resp.status_code == 200
    print("Health OK")

def test_predict():
    """Verify prediction endpoint correctly classifies a cat image"""
    files = {"file": open("data/raw/PetImages/Cat/cat.0.jpg","rb")}
    resp = requests.post("http://localhost:8000/predict", files=files)
    assert resp.status_code == 200
    print("Prediction OK", resp.json())

if __name__ == "__main__":
    test_health()
    test_predict()
```

**CI/CD Integration:**

Smoke tests are typically run:
- After deployment to staging/production
- As part of the CD pipeline health check
- Before routing production traffic to new deployments
- In Kubernetes readiness probes

**Troubleshooting:**

- **Connection Refused**: Ensure API server is running on port 8000
- **404 Not Found**: Check endpoint paths match API routes
- **500 Server Error**: Verify MLflow model is accessible and loaded
- **File Not Found**: Ensure test image exists at specified path

## Key Features

- ✅ **DVC Pipeline**: Reproducible training workflow with data versioning
- ✅ **MLflow Integration**: Experiment tracking and model registry
- ✅ **Model Validation Gate**: Automatic champion model promotion based on metrics
- ✅ **FastAPI Serving**: Production-ready REST API with auto-documentation
- ✅ **Prometheus Metrics**: Request monitoring and observability
- ✅ **Kubernetes Ready**: Scalable deployment with HPA and PDB
- ✅ **CI/CD Automation**: GitHub Actions for testing and deployment
- ✅ **Comprehensive Testing**: Unit tests for training and serving modules

## MLflow Tracking

Models are automatically tracked and versioned:

- **Tracking URI**: `http://localhost:5050`
- **Experiment**: `pet_adoptation_classification`
- **Registered Model**: `pet-classification-model`
- **Production Alias**: `champion` (promoted via validation gate)

Access MLflow UI at http://localhost:5050 to:
- Compare experiment runs
- View metrics and parameters
- Manage model versions
- Download artifacts

## Testing

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=training --cov=serving -v

# Run specific modules
pytest training/test/ -v
pytest serving/test/ -v

# Run specific test file
pytest training/test/test_train.py -v
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5050` | MLflow tracking server URL |
| `HOST_NAME` | `0.0.0.0` | API server host |
| `PORT` | `8000` | API server port |
