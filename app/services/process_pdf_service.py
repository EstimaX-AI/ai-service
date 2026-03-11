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


def _process_pdf_via_modal(file_path: str):
    """Offload PDF processing to Modal serverless worker.
    Tries GPU first, falls back to CPU if GPU is unavailable.
    """
    import modal
    import concurrent.futures

    start_time = time.time()

    # --- Try GPU first ---
    try:
        logger.info(f"[Modal] Trying GPU function 'process_pdf_job_gpu' in app 'ai-worker'")
        gpu_fn = modal.Function.from_name("ai-worker", "process_pdf_job_gpu")
        logger.info(f"[Modal] GPU function found, calling remote with file_path: {file_path}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(gpu_fn.remote, file_path)
            result = future.result(timeout=900)  # 15 minute timeout

        elapsed = time.time() - start_time
        logger.info(f"[Modal] GPU call completed in {elapsed:.1f}s. Status: {result.get('status')}")
        return result["status"], result["result"]

    except concurrent.futures.TimeoutError:
        elapsed = time.time() - start_time
        logger.error(f"[Modal] GPU call timed out after {elapsed:.1f}s")
    except Exception as e:
        logger.warning(f"[Modal] GPU function failed: {str(e)}, falling back to CPU")

    # --- Fallback to CPU ---
    try:
        logger.info(f"[Modal] Trying CPU function 'process_pdf_job_cpu' in app 'ai-worker'")
        cpu_fn = modal.Function.from_name("ai-worker", "process_pdf_job_cpu")
        logger.info(f"[Modal] CPU function found, calling remote with file_path: {file_path}")

        start_time = time.time()  # reset timer for CPU
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(cpu_fn.remote, file_path)
            result = future.result(timeout=900)

        elapsed = time.time() - start_time
        logger.info(f"[Modal] CPU call completed in {elapsed:.1f}s. Status: {result.get('status')}")
        return result["status"], result["result"]

    except concurrent.futures.TimeoutError:
        elapsed = time.time() - start_time
        raise TimeoutError(f"Modal CPU call timed out after {elapsed:.1f}s")
    except Exception as e:
        raise RuntimeError(f"Both GPU and CPU Modal functions failed. CPU error: {str(e)}")


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
