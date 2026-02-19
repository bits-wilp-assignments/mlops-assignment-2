from fastapi import FastAPI, UploadFile, File
from serving.src.inference.predictor import Predictor
from common.logger import get_logger

logger = get_logger(__name__)
app = FastAPI()
predictor = Predictor()

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"
    with open(file_path,"wb") as f:
        f.write(await file.read())
    result = predictor.predict(file_path)
    return result
