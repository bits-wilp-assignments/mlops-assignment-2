import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from serving.src.inference.predictor import Predictor
from serving.src.config.settings import MODEL_PATH, TEMP_DIR, HOST_NAME, PORT
from common.logger import get_logger

logger = get_logger(__name__)
app = FastAPI()
predictor = Predictor()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_PATH.split('/')[-1].replace('.h5', '')
    }

@app.put("/predict")
async def predict_put(file: UploadFile = File(...)):
    os.makedirs(TEMP_DIR, exist_ok=True)
    file_path = os.path.join(TEMP_DIR, file.filename)
    with open(file_path,"wb") as f:
        f.write(await file.read())
    result = predictor.predict(file_path)
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