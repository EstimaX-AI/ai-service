import logging
import traceback
import threading
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from utils.rabbitmq_client import rabbitmq_client
from fastapi import FastAPI, Response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Track consumer thread status
consumer_status = {
    "running": False,
    "started_at": None,
    "last_error": None,
    "thread_alive": False
}
_consumer_thread = None

def run_consumer():
    consumer_status["running"] = True
    consumer_status["started_at"] = datetime.now(timezone.utc).isoformat()
    logger.info("Consumer thread started")
    try:
        rabbitmq_client.start_consuming()
    except Exception as e:
        consumer_status["last_error"] = str(e)
        logger.error(f"Consumer error: {str(e)}")
        traceback.print_exc()
    finally:
        consumer_status["running"] = False
        logger.info("Consumer thread exited")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _consumer_thread
    logger.info("=== AI SERVICE STARTING ===")
    _consumer_thread = threading.Thread(target=run_consumer, daemon=True, name="rabbitmq-consumer")
    _consumer_thread.start()
    logger.info("Consumer thread launched")
    yield
    logger.info("=== AI SERVICE SHUTTING DOWN ===")
    rabbitmq_client.close()
    logger.info("RabbitMQ connection closed")

app = FastAPI(lifespan=lifespan)

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"message": "AI service"}

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    is_healthy = _consumer_thread is not None and _consumer_thread.is_alive()
    status_code = 200 if is_healthy else 503
    return Response(
        content='{"status": "healthy"}' if is_healthy else '{"status": "unhealthy"}',
        status_code=status_code,
        media_type="application/json"
    )

@app.get("/status")
def status():
    return {
        "service": "ai-service",
        "consumer_running": consumer_status["running"],
        "consumer_thread_alive": _consumer_thread.is_alive() if _consumer_thread else False,
        "started_at": consumer_status["started_at"],
        "last_error": consumer_status["last_error"],
        "uptime": str(datetime.now(timezone.utc)) if consumer_status["started_at"] else None
    }
