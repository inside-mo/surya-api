from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from PIL import Image
import fitz  # PyMuPDF for PDF handling
from io import BytesIO
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
import os

app = FastAPI()
recognition_predictor = RecognitionPredictor()
detection_predictor = DetectionPredictor()

API_KEY = os.getenv("API_KEY", "your-default-key")

@app.post("/ocr")
async def ocr(file: UploadFile = File(...), x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    content = await file.read()
    pdf = fitz.open(stream=content, filetype="pdf")
    results = []
    
    for page in pdf:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        predictions = recognition_predictor([img], det_predictor=detection_predictor)
        results.extend(predictions)
    
    return {"results": results}
