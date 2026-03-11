"""
Modal serverless worker for PDF symbol detection using YOLO.

Deploy:   modal deploy modal_worker.py
Test run: modal run modal_worker.py
"""

import modal

# ---------------------------------------------------------------------------
# Modal App & Images
# ---------------------------------------------------------------------------
app = modal.App("ai-worker")

# Shared dependencies (everything except torch)
_common_packages = [
    "ultralytics",
    "opencv-python-headless",
    "pymupdf",
    "numpy",
    "pillow",
    "requests",
]

# GPU image: installs CUDA-enabled torch
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision")  # defaults to CUDA build
    .pip_install(*_common_packages)
    .add_local_file(
        "app/models/best.pt",
        remote_path="/root/models/best.pt",
        copy=True,
    )
)

# CPU image: installs CPU-only torch (smaller, faster cold-start)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        extra_options="--index-url https://download.pytorch.org/whl/cpu",
    )
    .pip_install(*_common_packages)
    .add_local_file(
        "app/models/best.pt",
        remote_path="/root/models/best.pt",
        copy=True,
    )
)


# ---------------------------------------------------------------------------
# Shared processing logic
# ---------------------------------------------------------------------------
def _process_pdf(file_path: str) -> dict:
    """
    Core PDF processing logic shared by GPU and CPU functions.

    Args:
        file_path: URL (http/https) or absolute local path to the PDF.

    Returns:
        dict with keys: status ("success" | "error"), result (symbol counts or error info)
    """
    import os
    import tempfile
    import requests as req
    import cv2
    import numpy as np
    import torch
    from torchvision.ops import nms
    from ultralytics import YOLO
    import pymupdf as fitz
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Modal Worker] Using device: {device}")

    MODEL_PATH = "/root/models/best.pt"
    model = YOLO(MODEL_PATH)
    if device == "cuda":
        model.to("cuda")

    # ------------------------------------------------------------------
    # Download PDF if URL
    # ------------------------------------------------------------------
    tmp_path = None
    print(f"[Modal Worker] Starting PDF processing for: {file_path}")
    try:
        if file_path.startswith("http://") or file_path.startswith("https://"):
            print(f"[Modal Worker] Downloading PDF from URL...")
            resp = req.get(file_path, timeout=60)
            resp.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(resp.content)
            tmp.close()
            tmp_path = tmp.name
            local_path = tmp_path
            print(f"[Modal Worker] PDF downloaded: {len(resp.content)} bytes -> {tmp_path}")
        else:
            local_path = file_path
            print(f"[Modal Worker] Using local file: {local_path}")

        # ------------------------------------------------------------------
        # Sliding-window detection (same logic as pdf_reader.py)
        # ------------------------------------------------------------------
        def detect_symbols_in_image(
            mdl, img_np, window_size=640, stride=512,
            conf_threshold=0.3, iou_threshold=0.45,
        ):
            height, width = img_np.shape[:2]
            all_dets = []

            for y in range(0, height, stride):
                for x in range(0, width, stride):
                    y_end = min(y + window_size, height)
                    x_end = min(x + window_size, width)
                    window = img_np[y:y_end, x:x_end]

                    if window.shape[0] < window_size or window.shape[1] < window_size:
                        padded = np.zeros((window_size, window_size, 3), dtype=np.uint8)
                        padded[: window.shape[0], : window.shape[1]] = window
                        window = padded

                    window_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
                    window_pil = Image.fromarray(window_rgb)

                    results = mdl.predict(
                        window_pil,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        verbose=False,
                    )

                    for r in results:
                        for b in r.boxes:
                            xyxy = b.xyxy[0].cpu().numpy()
                            conf = float(b.conf[0].cpu())
                            cls_id = int(b.cls[0].cpu())
                            all_dets.append([
                                x + xyxy[0], y + xyxy[1],
                                x + xyxy[2], y + xyxy[3],
                                conf, cls_id,
                            ])

            if not all_dets:
                return {}

            boxes = torch.tensor([d[:4] for d in all_dets])
            scores = torch.tensor([d[4] for d in all_dets])
            keep = nms(boxes, scores, iou_threshold)
            kept_dets = [all_dets[i] for i in keep]

            counts = {}
            for d in kept_dets:
                cls_id = int(d[5])
                name = mdl.names.get(cls_id, f"class_{cls_id}")
                counts[name] = counts.get(name, 0) + 1
            return counts

        # ------------------------------------------------------------------
        # Process each page
        # ------------------------------------------------------------------
        doc = fitz.open(local_path)
        total_counts = {}
        dpi = 200
        print(f"[Modal Worker] PDF opened: {len(doc)} pages, dpi={dpi}")

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            zoom = dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.h, pix.w, pix.n
            )
            if pix.n == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            print(f"[Modal Worker] Processing page {page_idx + 1}/{len(doc)} (size: {pix.w}x{pix.h})...")
            page_counts = detect_symbols_in_image(model, img_np)
            for k, v in page_counts.items():
                total_counts[k] = total_counts.get(k, 0) + v

            print(f"[Modal Worker] Page {page_idx + 1} counts: {page_counts}")

        doc.close()

        print(f"[Modal Worker] Processing complete on {device}.")
        print(f"[Modal Worker] Final total counts:")
        for k, v in sorted(total_counts.items()):
            print(f"  {k}: {v}")

        return {"status": "success", "result": total_counts}

    except Exception as e:
        print(f"[Modal Worker] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "result": {"error": str(e)}}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            print(f"[Modal Worker] Cleaning up temp file: {tmp_path}")
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# GPU function (preferred — faster inference)
# ---------------------------------------------------------------------------
@app.function(image=gpu_image, timeout=3600, gpu="T4")
def process_pdf_job_gpu(file_path: str) -> dict:
    print("[Modal Worker] Running on GPU function (T4)")
    return _process_pdf(file_path)


# ---------------------------------------------------------------------------
# CPU function (fallback — no GPU needed, faster cold-start)
# ---------------------------------------------------------------------------
@app.function(image=cpu_image, timeout=3600)
def process_pdf_job_cpu(file_path: str) -> dict:
    print("[Modal Worker] Running on CPU function (fallback)")
    return _process_pdf(file_path)
