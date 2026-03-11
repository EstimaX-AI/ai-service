import os
import logging
import tempfile
import time
import requests
from core.config import Config
from inference.pdf_reader import process_pdf_for_symbols

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _download_pdf(url: str) -> str:
    logger.info(f"Downloading PDF from URL: {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(response.content)
    tmp.close()
    logger.info(f"PDF downloaded successfully, saved to: {tmp.name} ({len(response.content)} bytes)")
    return tmp.name


def _call_modal_function(func_name: str, file_path: str, label: str, max_retries: int = 2):
    """Call a single Modal function with retry for transient gRPC errors."""
    import modal

    logger.info(f"[Modal] Calling '{func_name}' ({label}) with file_path: {file_path}")
    fn = modal.Function.from_name("ai-worker", func_name)

    for attempt in range(1, max_retries + 1):
        start_time = time.time()
        try:
            result = fn.remote(file_path)
            elapsed = time.time() - start_time
            logger.info(f"[Modal] {label} call completed in {elapsed:.1f}s. Status: {result.get('status')}")
            return result["status"], result["result"]
        except (modal.exception.ConnectionError, TimeoutError) as e:
            elapsed = time.time() - start_time
            if attempt < max_retries:
                logger.warning(f"[Modal] {label} attempt {attempt}/{max_retries} failed after {elapsed:.1f}s: {str(e)}. Retrying...")
                time.sleep(3)
            else:
                logger.error(f"[Modal] {label} failed after {max_retries} attempts ({elapsed:.1f}s): {str(e)}")
                raise


def _process_pdf_via_modal(file_path: str):
    """Offload PDF processing to Modal serverless worker.
    Respects Config.MODAL_GPU: 'gpu', 'cpu', or 'auto' (try GPU, fallback CPU).
    """
    gpu_mode = Config.MODAL_GPU
    logger.info(f"[Modal] MODAL_GPU={gpu_mode}")

    # --- GPU only ---
    if gpu_mode == "gpu":
        return _call_modal_function("process_pdf_job_gpu", file_path, "GPU")

    # --- CPU only ---
    if gpu_mode == "cpu":
        return _call_modal_function("process_pdf_job_cpu", file_path, "CPU")

    # --- Auto: try GPU first, fallback to CPU ---
    try:
        return _call_modal_function("process_pdf_job_gpu", file_path, "GPU")
    except Exception as e:
        logger.warning(f"[Modal] GPU function failed: {str(e)}, falling back to CPU")

    return _call_modal_function("process_pdf_job_cpu", file_path, "CPU")


def process_pdf(user_id: str, job_id: str, file_path: str):
    logger.info(f"Processing PDF for user={user_id}, job={job_id}, file={file_path}")
    logger.info(f"USE_MODAL={Config.USE_MODAL}")

    # ── Modal path (serverless) ──────────────────────────────────────
    if Config.USE_MODAL:
        logger.info(f"[Modal] Offloading job {job_id} to Modal")
        try:
            status, result = _process_pdf_via_modal(file_path)
            logger.info(f"[Modal] Job {job_id} completed with status: {status}")
            return status, result
        except Exception as e:
            logger.error(f"[Modal] Failed to process PDF: {str(e)}", exc_info=True)
            return "error", {"error": str(e)}

    # ── Local path (existing behaviour) ──────────────────────────────
    model_path = os.path.join(BASE_DIR, "models", "best.pt")
    output_dir = os.path.join(BASE_DIR, "detections")
    tmp_path = None
    logger.info(f"[Local] Processing job {job_id} locally. Model: {model_path}")

    try:
        if file_path.startswith("http://") or file_path.startswith("https://"):
            logger.info(f"[Local] Downloading PDF from URL for job {job_id}")
            tmp_path = _download_pdf(file_path)
            local_path = tmp_path
        else:
            local_path = file_path
            logger.info(f"[Local] Using local file path: {local_path}")

        logger.info(f"[Local] Starting symbol detection for job {job_id}")
        flag, result = process_pdf_for_symbols(local_path, model_path=model_path, output_dir=output_dir, dpi=200)
        if flag:
            logger.info(f"[Local] Job {job_id} completed successfully. Result: {result}")
            return "success", result
        else:
            logger.warning(f"[Local] Job {job_id} failed: PDF processing returned False")
            return "error", {"error": "PDF processing failed"}
    except Exception as e:
        logger.error(f"[Local] Failed to process PDF for job {job_id}: {str(e)}", exc_info=True)
        return "error", {"error": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            logger.info(f"[Local] Cleaning up temp file: {tmp_path}")
            os.remove(tmp_path)
