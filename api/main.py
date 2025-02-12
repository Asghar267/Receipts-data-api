from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings
import google.generativeai as genai
import json
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
import fitz  # PyMuPDF
import logging
from typing import Dict, Any
import io

# Configuration
class Settings(BaseSettings):
    google_api_key: str
    ocr_eps: float = 50.0
    ocr_min_samples: int = 1
    response_mime_type: str = "application/json"
    
    class Config:
        env_file = ".env"

settings = Settings()

genai.configure(api_key=settings.google_api_key)

# Initialize OCR models once (heavy initialization)
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

app = FastAPI(title="Receipt Processing API", 
             description="API for processing receipts and extracting structured data",
             version="1.0.0")

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cluster_and_sort_text_blocks(text_blocks: list) -> list:
    """Cluster and sort text blocks to preserve layout structure."""
    if not text_blocks:
        return []
    
    try:
        centroids = np.array([(
            (box[0][0] + box[2][0]) / 2, 
            (box[0][1] + box[2][1]) / 2
        ) for (box, _, _) in text_blocks])
        
        clustering = DBSCAN(
            eps=settings.ocr_eps, 
            min_samples=settings.ocr_min_samples
        ).fit(centroids)
        
        labels = clustering.labels_
        clustered_blocks = {}
        
        for label, block in zip(labels, text_blocks):
            clustered_blocks.setdefault(label, []).append(block)
        
        # Sort clusters by vertical position
        sorted_clusters = sorted(
            clustered_blocks.items(),
            key=lambda item: min(min(pt[1] for pt in blk[0]) for blk in item[1])
        )
        
        sorted_blocks = []
        for label, blocks in sorted_clusters:
            blocks.sort(key=lambda blk: min(pt[0] for pt in blk[0]))
            sorted_blocks.extend(blocks)
            
        return sorted_blocks
    except Exception as e:
        logger.error(f"Clustering error: {str(e)}")
        return text_blocks

def extract_text_with_layout(image: Image.Image) -> str:
    """Extract text from image with layout preservation."""
    try:
        result = ocr_model.ocr(np.array(image), cls=True, det=True, rec=True)
        text_blocks = []
        
        for line in result[0]:
            text = line[1][0]
            box = np.array(line[0], dtype=np.int32)
            text_blocks.append((box, text, line[1][1]))
        
        sorted_blocks = cluster_and_sort_text_blocks(text_blocks)
        return "\n".join([block[1] for block in sorted_blocks])
    except Exception as e:
        logger.error(f"OCR Error: {str(e)}")
        raise

def process_pdf(pdf_path: str) -> str:
    """Process PDF document and extract structured text."""
    try:
        pdf_document = fitz.open(pdf_path)
        extracted_text = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(dpi=450)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = extract_text_with_layout(image)
            extracted_text.append(f"--- Page {page_num+1} ---\n{page_text}")
        
        return "\n".join(extracted_text)
    except Exception as e:
        logger.error(f"PDF Processing Error: {str(e)}")
        raise
    finally:
        if 'pdf_document' in locals():
            pdf_document.close()

def extract_structured_data(raw_text: str) -> Dict[str, Any]:
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""Extract following details from receipt text:
        - Sender Name
        - Sender Account Number
        - Receiver Name
        - Receiver Account Number
        - Transaction Date (YYYY-MM-DD)
        - Transaction Amount (numeric)
        - Status [Success, Pending, Failed]
        Return JSON format:
        {{"sender_name": "", "sender_account": "", "receiver_name": "", 
        "receiver_account": "", "transaction_date": "", "amount": 0.0, "status": ""}}
        
        Text: {raw_text}"""
        
        response = model.generate_content(prompt)
        response_text = response.text.strip().strip('```json').strip('```')
         
        return json.loads(response_text)
    except json.JSONDecodeError as json_error:
        logger.error(f"Failed to parse Gemini response: {str(json_error)}")
        return {"error": f"Failed to parse AI response: {str(json_error)} - {response.text}"}
    except Exception as e:
        logger.error(f"AI Error: {str(e)}")
        return {"error": str(e)}

 
@app.post("/process-image", summary="Process image receipt")
async def process_image_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
# async def process_image_endpoint(file: UploadFile = File(...), settings: Settings = Depends(get_settings), ocr_model: PaddleOCR = Depends(get_ocr_model)) -> Dict[str, Any]:

    """Process image receipt and extract structured data."""
    try:
        content = await file.read()
        if len(content) > 3 * 1024 * 1024:  # 3MB limit
            raise HTTPException(400, "File too large. Maximum size is 10MB.")
            
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(400, "Invalid file type. Only JPEG and PNG are supported.")
        
        image = Image.open(io.BytesIO(content)).convert('RGB')
        raw_text = extract_text_with_layout(image)
        
       
        if not raw_text.strip():
            raise HTTPException(400, "No text extracted from image")
        
        structured_data = extract_structured_data(raw_text)
        if "error" in structured_data:
            raise HTTPException(500, structured_data["error"])
            
        return JSONResponse(content=structured_data)
        
    except Exception as e:
        logger.error(f"Image Processing Failed: {str(e)}")
        raise HTTPException(500, str(e))

 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)