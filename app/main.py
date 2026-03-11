import logging
import traceback
from utils.rabbitmq_client import rabbitmq_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting AI service")
    try:
        rabbitmq_client.start_consuming()
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        traceback.print_exc()
    finally:
        rabbitmq_client.close()
        logger.info("AI service stopped")

if __name__ == "__main__":
    main()
