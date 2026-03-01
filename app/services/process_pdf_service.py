import os
import tempfile
import shutil
from fastapi import UploadFile
from inference.pdf_reader import process_pdf_for_symbols

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def process_pdf(user_id: str, job_id: str, pdf_file: UploadFile):
    model_path = os.path.join(BASE_DIR, "models", "best.pt")
    output_dir = os.path.join(BASE_DIR, "detections")

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(pdf_file.file, tmp)
        tmp_path = tmp.name

    try:
        result = process_pdf_for_symbols(tmp_path, model_path=model_path, output_dir=output_dir, dpi=200)
    finally:
        # Clean up the temp file
        os.unlink(tmp_path)

    return result