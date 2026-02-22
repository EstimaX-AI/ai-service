```
ai-service/
│
├── app/
│   ├── main.py                      # Worker entrypoint
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── exceptions.py
│   │
│   ├── messaging/
│   │   ├── connection.py
│   │   ├── consumer.py              # Consume job.created
│   │   ├── publisher.py             # Publish job.completed/failed
│   │   └── schemas.py               # Message validation
│   │
│   ├── inference/
│   │   ├── model_loader.py          # Load YOLO model once
│   │   ├── predictor.py             # Run forward pass
│   │   ├── postprocess.py           # Threshold + NMS
│   │   ├── annotate.py              # Draw bounding boxes
│   │   └── transforms.py            # Preprocessing
│   │
│   ├── services/
│   │   ├── job_processor.py         # Full orchestration
│   │   ├── downloader.py            # Download original image
│   │   ├── uploader.py              # Upload annotated image
│   │   └── result_builder.py        # Build result JSON
│   │
│   ├── models/                      # Model weights
│   │   └── yolo_v8_symbol.pt
│   │
│   └── utils/
│       ├── image_utils.py
│       └── validation.py
│
├── tests/
│   ├── test_inference.py
│   ├── test_annotation.py
│   └── test_messaging.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.dev
└── README.md
```
# Blueprint AI Inference Service

## RabbitMQ Messaging Contract (Image Enabled) – v1.0

---

# 1. Overview

This document defines the asynchronous contract between:

* Backend (Producer + Result Consumer)
* AI Inference Service (Consumer + Result Producer)

Protocol: RabbitMQ (AMQP 0-9-1)
Message Format: JSON
Delivery Mode: Persistent

---

# 2. Exchange & Queues

Exchange:

* blueprint.ai.exchange (direct)

Queues:

* blueprint.ai.jobs      (AI consumes)
* blueprint.ai.results   (Backend consumes)

Routing Keys:

* job.created
* job.completed
* job.failed

---

# 3. Backend → AI Message

## Routing Key

job.created

## Message Body

```json
{
  "event": "JOB_CREATED",
  "job_id": "uuid",
  "file_url": "https://storage/original/uuid.png",
  "confidence_threshold": 0.25,
  "model_version": "yolo_v8_symbol",
  "timestamp": "2026-02-22T10:25:45Z"
}
```

AI Responsibilities:

* Download original image
* Run inference
* Generate annotated image
* Upload annotated image to storage
* Publish completion message
* Acknowledge only after success

---

# 4. AI → Backend Success Message

## Routing Key

job.completed

## Message Body

```json
{
  "event": "JOB_COMPLETED",
  "job_id": "uuid",
  "model_version": "yolo_v8_symbol_2.1",
  "processing_time_ms": 4820,
  "image_size": {
    "width": 2480,
    "height": 3508
  },
  "annotated_image_url": "https://storage/annotated/uuid.png",
  "detections": [
    {
      "label": "valve",
      "confidence": 0.91,
      "bbox": {
        "x1": 100,
        "y1": 200,
        "x2": 180,
        "y2": 280
      }
    }
  ],
  "timestamp": "2026-02-22T10:26:09Z"
}
```

Backend Responsibilities:

* Persist detections
* Save annotated_image_url
* Update status → COMPLETED
* Aggregate symbol counts

---

# 5. AI → Backend Failure Message

## Routing Key

job.failed

## Message Body

```json
{
  "event": "JOB_FAILED",
  "job_id": "uuid",
  "error": "MODEL_INFERENCE_FAILED",
  "reason": "CUDA out of memory",
  "timestamp": "2026-02-22T10:26:09Z"
}
```

Backend Action:

* Update job status → FAILED
* Store failure reason

---

# 6. Reliability Rules

* Manual ACK required
* Dead Letter Queue recommended
* Idempotent result handling required
* Timeout handling required
* Retry limit recommended (max 3)

---

# 7. Status Flow

Backend:  PENDING → QUEUED
AI:       PROCESSING
Backend:  COMPLETED / FAILED

---

End of AI RabbitMQ Messaging Specification
