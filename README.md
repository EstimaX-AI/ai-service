# 🏗️ EstimaX AI Service

> **Serverless PDF symbol detection** powered by YOLO, RabbitMQ, and Modal.
> Processes construction blueprint PDFs, detects engineering symbols, and returns aggregated counts — locally or in the cloud.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EstimaX Backend                                 │
│                                                                         │
│   User uploads PDF ──► Stores in Supabase S3 ──► Publishes to RabbitMQ │
│                                                       │                 │
│   Result ◄── Consumes from result_queue ◄─────────────┼─────────┐      │
│   Notification ◄── Consumes from notification_queue ◄─┼────┐    │      │
└───────────────────────────────────────────────────────┼────┼────┼──────┘
                                                        │    │    │
                                              ┌─────────▼────┴────┴──────┐
                                              │     AI Service (this)     │
                                              │                          │
                                              │  RabbitMQ Consumer       │
                                              │       │                  │
                                              │       ▼                  │
                                              │  process_pdf_service     │
                                              │       │                  │
                                              │       ├─► Local (YOLO)   │
                                              │       │                  │
                                              │       └─► Modal Cloud    │
                                              │            ├─ GPU (T4)   │
                                              │            └─ CPU        │
                                              │                          │
                                              │  Publishes results to:   │
                                              │  • result_queue          │
                                              │  • notification_queue    │
                                              └──────────────────────────┘
```

---

## 📂 Project Structure

```
Ai_service/
├── modal_worker.py              # Modal serverless functions (GPU + CPU)
├── requirements.txt
├── Dockerfile
│
└── app/
    ├── main.py                  # FastAPI app + RabbitMQ consumer thread
    │
    ├── core/
    │   └── config.py            # Environment config (RabbitMQ, Modal, GPU)
    │
    ├── inference/
    │   └── pdf_reader.py        # YOLO sliding-window detection on PDF pages
    │
    ├── services/
    │   └── process_pdf_service.py  # Orchestrator: Modal vs Local routing
    │
    ├── utils/
    │   └── rabbitmq_client.py   # Thread-safe RabbitMQ consumer + publisher
    │
    ├── models/
    │   └── best.pt              # YOLO model weights
    │
    ├── detections/              # Local output: per-page detection images
    ├── .env.dev                 # Environment variables
    └── .env.example             # Template
```

---

## ⚡ How It Works — Full Workflow

### Step 1 → Job arrives via RabbitMQ

The backend publishes a message to the `ai_jobs` queue:

```json
{
  "user_id": "12345",
  "job_id": "c77c5e69-a262-4322-b172-fcd052221591",
  "file_path": "https://storage.supabase.co/.../blueprint.pdf?token=..."
}
```

### Step 2 → Consumer picks it up

`main.py` runs a **daemon consumer thread** that listens on `ai_jobs`. When a message arrives:

```
Consumer Thread ──► rabbitmq_client.process_message()
                         │
                         ├── Parses payload (user_id, job_id, file_path)
                         ├── Calls process_pdf_service.process_pdf()
                         ├── Publishes result to result_queue
                         ├── Publishes notification to notification_queue
                         └── ACKs the original message
```

### Step 3 → Processing (Local or Modal)

Controlled by two env vars:

| Variable     | Value   | Behavior                                  |
|-------------|---------|-------------------------------------------|
| `USE_MODAL` | `true`  | Offload to Modal serverless               |
| `USE_MODAL` | `false` | Process locally with YOLO                 |
| `MODAL_GPU` | `auto`  | Try GPU → fallback to CPU (Modal only)    |
| `MODAL_GPU` | `gpu`   | GPU only — fail if unavailable            |
| `MODAL_GPU` | `cpu`   | CPU only — faster cold-start              |

#### Local Path
1. Downloads PDF if it's a URL
2. Loads YOLO model from `app/models/best.pt`
3. Converts each PDF page to an image (200 DPI)
4. Runs sliding-window detection (640×640 windows, stride 512)
5. Applies NMS to remove duplicate detections
6. Returns aggregated symbol counts

#### Modal Path
1. Resolves the Modal function (`process_pdf_job_gpu` or `process_pdf_job_cpu`)
2. Calls `.remote(file_path)` — Modal handles container provisioning
3. Inside the Modal container: same YOLO pipeline runs with the bundled model
4. Automatic retry (2 attempts) on transient gRPC `ConnectionError`

### Step 4 → Result published

Two messages are published back:

**Result Queue** (`result_queue`):
```json
{
  "user_id": "12345",
  "job_id": "c77c5e69-a262-4322-b172-fcd052221591",
  "status": "success",
  "result": "{\"valve\": 12, \"pump\": 3, \"motor\": 7}",
  "created_at": "2026-03-11T22:57:54"
}
```

**Notification Queue** (`notification_queue`):
```json
{
  "user_id": "12345",
  "job_id": "c77c5e69-a262-4322-b172-fcd052221591",
  "message": "PDF processed successfully",
  "status": "success",
  "created_at": "2026-03-11T22:57:54"
}
```

> The `result` field is a **JSON-encoded string** of symbol name → count.

### Error Response

On failure, `status` is `"error"` and `result` contains:
```json
{
  "error": "Description of what went wrong"
}
```

---

## 🚀 Quick Start

### 1. Setup

```bash
cd Ai_service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp app/.env.example app/.env.dev
# Edit app/.env.dev with your credentials
```

### 3. Run Locally

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### 4. Deploy Modal Worker (optional)

```bash
modal setup          # First time only — authenticate
modal deploy modal_worker.py
```

Then set in `.env.dev`:
```
USE_MODAL=true
MODAL_GPU=cpu        # or gpu / auto
```

---

## 🐳 Docker

```bash
docker build -t estimax-ai .
docker run -p 8000:8000 --env-file app/.env.dev estimax-ai
```

---

## 🔧 Environment Variables

| Variable              | Default              | Description                                |
|----------------------|----------------------|--------------------------------------------|
| `MQ_HOST`            | `localhost`          | RabbitMQ host                              |
| `MQ_PORT`            | `5672`               | RabbitMQ port                              |
| `MQ_USER`            | `guest`              | RabbitMQ username                          |
| `MQ_PASSWORD`        | `guest`              | RabbitMQ password                          |
| `MQ_VHOST`           | `/`                  | RabbitMQ virtual host                      |
| `PDF_QUEUE`          | `ai_jobs`            | Queue to consume jobs from                 |
| `RESULT_QUEUE`       | `result_queue`       | Queue to publish results to                |
| `NOTIFICATION_QUEUE` | `notification_queue` | Queue to publish notifications to          |
| `USE_MODAL`          | `false`              | Enable Modal serverless inference          |
| `MODAL_GPU`          | `auto`               | GPU mode: `auto` / `gpu` / `cpu`           |

---

## 🔍 YOLO Detection Pipeline

The detection engine uses a **sliding-window** approach optimized for large blueprint images:

```
PDF Page (200 DPI)
    │
    ▼
Rasterize via PyMuPDF
    │
    ▼
Sliding Window (640×640, stride 512)
    │
    ▼
YOLO Inference per window
    │
    ▼
Aggregate all detections
    │
    ▼
Non-Maximum Suppression (NMS)
    │
    ▼
Symbol counts per class
```

**Parameters:**
- Window size: `640×640`
- Stride: `512` (128px overlap for edge detection)
- Confidence threshold: `0.3`
- IoU threshold: `0.45`
- DPI: `200`

---

## 🌐 API Endpoints

| Method       | Path      | Description                       |
|-------------|-----------|-----------------------------------|
| `GET/HEAD`  | `/`       | Service info                      |
| `GET/HEAD`  | `/health` | Health check (consumer alive?)    |
| `GET`       | `/status` | Detailed consumer thread status   |

---

## 📡 RabbitMQ Queues

```
ai_jobs               ← Backend publishes jobs
result_queue          ← AI publishes processing results
notification_queue    ← AI publishes status notifications
```

All queues are **durable** with **persistent** messages and **manual ACK**.

---

## ☁️ Modal Functions

| Function               | Image       | GPU  | Description              |
|------------------------|-------------|------|--------------------------|
| `process_pdf_job_gpu`  | CUDA torch  | T4   | Faster inference         |
| `process_pdf_job_cpu`  | CPU torch   | None | Faster cold-start        |

Both share the same detection logic. The GPU image includes CUDA-enabled PyTorch, while the CPU image uses the lightweight CPU-only build for faster container startup.

---

## 🔄 Status Flow

```
Backend:   PENDING ──► QUEUED ──────────────────────────► COMPLETED / FAILED
                          │                                      ▲
                          ▼                                      │
AI Service:          CONSUMING ──► PROCESSING ──► PUBLISHING ────┘
                                      │
                                      ├── Local (YOLO)
                                      └── Modal (GPU / CPU)
```
