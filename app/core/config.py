import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env.dev"))

class Config:
    RABBIT_HOST: str = os.getenv("MQ_HOST", "localhost")
    RABBIT_PORT: int = int(os.getenv("MQ_PORT", "5672"))
    RABBIT_USER: str = os.getenv("MQ_USER", "guest")
    RABBITMQ_PASSWORD: str = os.getenv("MQ_PASSWORD", "guest")
    RABBIT_VHOST: str = os.getenv("MQ_VHOST", "/")

    ai_jobs: str = os.getenv("PDF_QUEUE", "ai_jobs")
    result_queue: str = os.getenv("RESULT_QUEUE", "result_queue")
    notification_queue: str = os.getenv("NOTIFICATION_QUEUE", "notification_queue")
