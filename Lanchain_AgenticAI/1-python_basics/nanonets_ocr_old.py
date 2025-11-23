"""
Synchronous version of the FastAPI application for the OCR service.
This file uses the original "request-and-wait" model for testing purposes,
without any background task or job polling logic.
"""
# ================= FastAPI and Standard Libraries =================
import io
import os
import json
import tempfile
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pdf2image import convert_from_bytes

# ================== Transformers / Model Inference ================
from transformers import AutoProcessor, AutoModelForImageTextToText

# =================== Custom Prompts ================================
from prompts import RESUME_PROMPT, EILIFY_PROMPT

# =================== Helper Modules ================================
from helpers.helper_resume import ocr_resume, clean_zip_from_resume_entry
from helpers.helper_eilify import ocr_eilify
from helpers.helper_smalter import (
    CniLanguage, SmalterDocumentType,
    get_document_creation_date_sync,
    clean_json_output,
    ocr_smalter_helper,
    run_bulk_ocr_task # We will use a modified version of this concept
)

# ***********MODEL AND APP SETUP***********
MODEL_PATH = "nanonets/Nanonets-OCR-s"
MODEL = None
PROCESSOR = None

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Loads the OCR model and processor on application startup."""
    global MODEL, PROCESSOR
    MODEL = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, device_map="auto")
    MODEL.eval()
    PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH)
    print("Model and processor loaded successfully.")
    yield
    print("Application shutdown complete.")

app = FastAPI(lifespan=lifespan)

# ***********API ENDPOINTS (SYNCHRONOUS)***********

@app.post("/ocr/resume")
async def extract_resume_fields(file: UploadFile = File(...)):
    """Synchronous endpoint to perform OCR on a single resume file."""
    try:
        content = await file.read()
        creation_date = get_document_creation_date_sync(file.filename, content)
        file_extension = os.path.splitext(file.filename)[1]
        file_type = file_extension.replace(".", "").upper() if file_extension else "UNKNOWN"

        if file_type == "PDF": images = convert_from_bytes(content, dpi=300)
        elif file_type in ["PNG", "JPG", "JPEG"]: images = [Image.open(io.BytesIO(content)).convert("RGB")]
        else: raise HTTPException(status_code=415, detail=f"Unsupported file type: {file_type}.")
        if not images: raise HTTPException(status_code=422, detail="Could not process file into an image.")

        raw_output = ocr_resume(images[0], MODEL, PROCESSOR, RESUME_PROMPT, max_new_tokens=15000)
        cleaned_json_string = clean_zip_from_resume_entry(raw_output)
        ocr_result = json.loads(cleaned_json_string)

        file_metadata = {"file_url": "/ocr/resume", "file_type": file_type, "file_creation_date": creation_date}
        details = [ocr_result]
        final_response = {"file_metadata": file_metadata, "details": details}
        return JSONResponse(content=final_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/ocr/smalter/{image_type}")
async def ocr_smalter_sync(
    image_type: SmalterDocumentType,
    file: UploadFile = File(...),
    lang: Optional[CniLanguage] = Form(None),
):
    """
    Synchronous endpoint that accepts a single document OR a ZIP file.
    All processing is done before returning the response.
    """
    is_cni = image_type in [SmalterDocumentType.CNI_FRONT, SmalterDocumentType.CNI_BACK]
    if is_cni and not lang:
        raise HTTPException(status_code=400, detail="The 'lang' parameter is required for CNI documents.")

    file_bytes = await file.read()

    # --- LOGIC DISPATCHER: Check for ZIP file ---
    if file.filename.lower().endswith('.zip'):
        supported_bulk_types = [SmalterDocumentType.CNI_FRONT, SmalterDocumentType.CNI_BACK, SmalterDocumentType.WORK_CONTRACT]
        if image_type not in supported_bulk_types:
            raise HTTPException(status_code=400, detail=f"Bulk processing for '{image_type.value}' is not supported.")
        
        # Directly call a synchronous version of the bulk processor
        try:
            # Note: This is a synchronous call. The client will wait.
            bulk_results = run_bulk_ocr_task_sync(file_bytes, image_type, MODEL, PROCESSOR, lang)
            return JSONResponse(content=bulk_results)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Bulk processing failed: {str(e)}")

    else: # Process a single file
        try:
            creation_date = get_document_creation_date_sync(file.filename, file_bytes)
            file_extension = os.path.splitext(file.filename)[1]
            file_type = file_extension.replace('.', '').upper() if file_extension else "UNKNOWN"

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name
            
            try:
                ocr_result = ocr_smalter_helper(tmp_path, image_type, MODEL, PROCESSOR, lang=lang)
            finally:
                os.remove(tmp_path)

            file_metadata = {"file_url": f"/ocr/smalter/{image_type.value}", "file_type": file_type, "file_creation_date": creation_date}
            if is_cni: file_metadata["language"] = lang.value
            
            details = ocr_result if isinstance(ocr_result, list) else [ocr_result]
            final_response = {"file_metadata": file_metadata, "details": details}
            return JSONResponse(content=final_response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/eilify")
async def elify_ocr(image: UploadFile = File(...)):
    """Synchronous endpoint for the 'Eilify' document type."""
    try:
        image_bytes = await image.read()
        creation_date = get_document_creation_date_sync(image.filename, image_bytes)
        file_extension = os.path.splitext(image.filename)[1]
        file_type = file_extension.replace('.', '').upper() if file_extension else "UNKNOWN"
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        output_text = ocr_eilify(image_pil, MODEL, PROCESSOR, EILIFY_PROMPT)
        ocr_result = clean_json_output(output_text)

        file_metadata = {"file_url": "/ocr/eilify", "file_type": file_type, "file_creation_date": creation_date}
        details = [ocr_result]
        final_response = {"file_metadata": file_metadata, "details": details}
        return JSONResponse(content=final_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}") from e

# Note: This is a new synchronous helper for bulk processing.
# It should be placed here or in the helper_smalter.py file.
def run_bulk_ocr_task_sync(zip_bytes: bytes, image_type: SmalterDocumentType, model, processor, lang: Optional[CniLanguage]):
    """
    Synchronous version of the bulk OCR task for testing.
    """
    all_results = []
    with tempfile.TemporaryDirectory() as temp_dir:
        with io.BytesIO(zip_bytes) as zip_buffer:
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        for root, _, files in os.walk(temp_dir):
            for filename in files:
                supported_extensions = ('.png', '.jpg', '.jpeg', '.pdf')
                if not filename.lower().endswith(supported_extensions): continue
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'rb') as f: file_bytes_for_meta = f.read()
                    creation_date = get_document_creation_date_sync(filename, file_bytes_for_meta)
                    file_extension = os.path.splitext(filename)[1]
                    file_type = file_extension.replace('.', '').upper() if file_extension else "UNKNOWN"
                    is_cni = image_type in [SmalterDocumentType.CNI_FRONT, SmalterDocumentType.CNI_BACK]
                    
                    ocr_result = ocr_smalter_helper(file_path, image_type, model, processor, lang=lang)
                    
                    file_metadata = {"source_file": filename, "file_url": f"/ocr/smalter/{image_type.value}", "file_type": file_type, "file_creation_date": creation_date}
                    if is_cni: file_metadata["language"] = lang.value
                    details = ocr_result if isinstance(ocr_result, list) else [ocr_result]
                    final_response = {"file_metadata": file_metadata, "details": details}
                    all_results.append(final_response)
                except Exception as e:
                    all_results.append({"file_metadata": {"source_file": filename, "error": True}, "details": [str(e)]})
    return all_results
