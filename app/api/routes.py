from fastapi import APIRouter
from services.process_pdf_service import process_pdf
from fastapi import UploadFile, File
router = APIRouter()

@router.get("/health")
def read_root():
    return {"status": "ok"}

@router.post("/analyse_pdf")
def analyse_pdf_api(user_id: str,job_id:str, pdf_file: UploadFile = File(...)):
    return process_pdf(user_id,job_id, pdf_file)