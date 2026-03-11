import logging
import traceback
import threading
from contextlib import asynccontextmanager
from utils.rabbitmq_client import rabbitmq_client
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_consumer():
    try:
        rabbitmq_client.start_consuming()
    except Exception as e:
        logger.error(f"Consumer error: {str(e)}")
        traceback.print_exc()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI service")
    thread = threading.Thread(target=run_consumer, daemon=True)
    thread.start()
    yield
    rabbitmq_client.close()
    logger.info("AI service stopped")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "AI service"}
