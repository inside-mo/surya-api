from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from PIL import Image
from io import BytesIO
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
import os

app = FastAPI()
recognition_predictor = RecognitionPredictor()
detection_predictor = DetectionPredictor()

API_KEY = os.getenv("API_KEY", "default-key")

@app.post("/ocr")
async def ocr(file: UploadFile = File(...), x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, message="Invalid API key")
    
    image = Image.open(BytesIO(await file.read()))
    predictions = recognition_predictor([image], det_predictor=detection_predictor)
    return predictions[0]
