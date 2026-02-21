import os
import time
import uvicorn
from fastapi import FastAPI, UploadFile, File
from prometheus_client import make_asgi_app
from serving.src.inference.predictor import Predictor
from serving.src.config.settings import TEMP_DIR, HOST_NAME, PORT
from serving.monitoring.metrics import RequestMetrics
from common.logger import get_logger

logger = get_logger(__name__)
app = FastAPI()

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Load model from MLflow registry with 'champion' alias
predictor = Predictor(model_alias="champion")
metrics = RequestMetrics()

@app.get("/health")
def health():
    start_time = time.time()
    health_response = {
        "status": "ok",
        "model_name": predictor.model_name,
        "model_alias": predictor.model_alias,
    }
    metrics.log_request(time.time() - start_time)
    return health_response

@app.put("/predict")
async def predict_put(file: UploadFile = File(...)):
    start_time = time.time()
    os.makedirs(TEMP_DIR, exist_ok=True)
    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path,"wb") as f:
        f.write(await file.read())
    result = predictor.predict(file_path)
    metrics.log_request(time.time() - start_time)
    return result


# Local Run
if __name__ == "__main__":
    logger.info(f"Starting Pet Adoption Classification API on {HOST_NAME}:{PORT}...")
    uvicorn.run(
        app,
        host=HOST_NAME,
        port=PORT,
        reload=False,
    )