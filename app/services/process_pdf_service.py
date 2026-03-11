import os
import logging
import tempfile
import requests
from inference.pdf_reader import process_pdf_for_symbols

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _download_pdf(url: str) -> str:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(response.content)
    tmp.close()
    return tmp.name

def process_pdf(user_id: str, job_id: str, file_path: str):
    model_path = os.path.join(BASE_DIR, "models", "best.pt")
    output_dir = os.path.join(BASE_DIR, "detections")
    tmp_path = None

    try:
        if file_path.startswith("http://") or file_path.startswith("https://"):
            logger.info(f"Downloading PDF from URL for job {job_id}")
            tmp_path = _download_pdf(file_path)
            local_path = tmp_path
        else:
            local_path = file_path

        flag, result = process_pdf_for_symbols(local_path, model_path=model_path, output_dir=output_dir, dpi=200)
        if flag:
            return "success", result
        else:
            return "error", {"error": "PDF processing failed"}
    except Exception as e:
        logger.error(f"Failed to process PDF: {str(e)}")
        return "error", {"error": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
