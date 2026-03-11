import json
import logging
import pika
import socket
import time
import threading
from datetime import datetime
from services import process_pdf_service

from core.config import Config

logging.getLogger("pika").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class RabbitMQClient:
    def __init__(self):
        self._connection = None
        self._channel = None
        self._publish_lock = threading.Lock()

    def _get_connection_params(self):
        ipv4_host = socket.getaddrinfo(Config.RABBIT_HOST, None, socket.AF_INET)[0][4][0]
        return pika.ConnectionParameters(
            host=ipv4_host,
            port=Config.RABBIT_PORT,
            virtual_host=Config.RABBIT_VHOST,
            credentials=pika.PlainCredentials(Config.RABBIT_USER, Config.RABBITMQ_PASSWORD),
            socket_timeout=30,
            heartbeat=1800,
            blocked_connection_timeout=300
        )

    def _connect(self):
        """Get or create the publish connection. Caller MUST hold _publish_lock."""
        if self._connection is None or self._connection.is_closed:
            logger.info("Creating new publish connection to RabbitMQ")
            self._connection = pika.BlockingConnection(self._get_connection_params())
            self._channel = self._connection.channel()
            self._channel.confirm_delivery()

            self._channel.queue_declare(queue=Config.ai_jobs, durable=True)
            self._channel.queue_declare(queue=Config.result_queue, durable=True)
            self._channel.queue_declare(queue=Config.notification_queue, durable=True)

        if self._channel is None or self._channel.is_closed:
            self._channel = self._connection.channel()
            self._channel.confirm_delivery()

        return self._channel

    def close(self):
        if self._channel and not self._channel.is_closed:
            self._channel.close()
        if self._connection and not self._connection.is_closed:
            self._connection.close()

    def publish_to_result_queue(self, user_id: int, job_id: str, result: dict, status: str):
        max_retries = 3
        message = {
            "user_id": user_id,
            "job_id": job_id,
            "status": status,
            "result": json.dumps(result),
            "created_at": datetime.now().isoformat()
        }

        for attempt in range(max_retries):
            try:
                with self._publish_lock:
                    # Force reconnect if stale (e.g. after long Modal call)
                    if self._connection and self._connection.is_closed:
                        logger.warning("Publish connection was closed, reconnecting...")
                        self._connection = None
                    channel = self._connect()
                    channel.basic_publish(
                        exchange="",
                        routing_key=Config.result_queue,
                        body=json.dumps(message),
                        properties=pika.BasicProperties(delivery_mode=2)
                    )
                logger.info(f"Published to result queue: {job_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to publish to result queue (attempt {attempt + 1}/{max_retries}): {str(e)}")
                with self._publish_lock:
                    self._connection = None
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to publish to result queue after {max_retries} attempts")
        return False

    def publish_to_notification_queue(self, user_id: int, job_id: str, message: str, status: str):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self._publish_lock:
                    if self._connection and self._connection.is_closed:
                        logger.warning("Publish connection was closed, reconnecting...")
                        self._connection = None
                    channel = self._connect()
                    channel.basic_publish(
                        exchange="",
                        routing_key=Config.notification_queue,
                        body=json.dumps({
                            "user_id": user_id,
                            "job_id": job_id,
                            "message": message,
                            "status": status,
                            "created_at": datetime.now().isoformat()
                        }),
                        properties=pika.BasicProperties(delivery_mode=2)
                    )
                logger.info(f"Published to notification queue: {job_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to publish to notification queue (attempt {attempt + 1}/{max_retries}): {str(e)}")
                with self._publish_lock:
                    self._connection = None
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to publish to notification queue after {max_retries} attempts")
        return False

    def process_message(self, body: bytes):
        if not body:
            logger.warning("Empty message dropped")
            return False
        body_str = body.decode("utf-8")

        try:
            payload = json.loads(body_str)
            user_id = payload["user_id"]
            job_id = payload["job_id"]
            file_path = payload.get("file_path") or payload["pdf_url"]
            status, result = process_pdf_service.process_pdf(user_id, job_id, file_path)
            self.publish_to_result_queue(user_id, job_id, result, status)
            self.publish_to_notification_queue(user_id, job_id, "PDF processed successfully", status)
            return True
        except Exception as e:
            logger.error(f"Failed to process message: {str(e)}")
            payload_data = json.loads(body_str) if body_str else {}
            self.publish_to_result_queue(
                payload_data.get("user_id"), payload_data.get("job_id"), {}, "error"
            )
            self.publish_to_notification_queue(
                payload_data.get("user_id"), payload_data.get("job_id"), "PDF processing failed", "error"
            )
            return False

    def start_consuming(self):
        while True:
            consume_connection = None
            try:
                logger.info("Connecting to RabbitMQ...")
                consume_connection = pika.BlockingConnection(self._get_connection_params())
                consume_channel = consume_connection.channel()
                consume_channel.queue_declare(queue=Config.ai_jobs, durable=True)
                consume_channel.basic_qos(prefetch_count=1)

                def callback(ch, method, properties, body):
                    job_id = json.loads(body).get('job_id', 'unknown')
                    logger.info(f"Received message for job: {job_id}")

                    def process():
                        delivery_tag = method.delivery_tag
                        try:
                            success = self.process_message(body)
                            if success:
                                logger.info(f"Job {job_id} processed successfully, sending ACK")
                                try:
                                    consume_connection.add_callback_threadsafe(
                                        lambda: ch.basic_ack(delivery_tag=delivery_tag)
                                    )
                                except Exception as ack_err:
                                    logger.error(f"Failed to ACK job {job_id}: {str(ack_err)}")
                            else:
                                logger.warning(f"Job {job_id} processing failed, sending NACK")
                                try:
                                    consume_connection.add_callback_threadsafe(
                                        lambda: ch.basic_nack(delivery_tag=delivery_tag, requeue=False)
                                    )
                                except Exception as nack_err:
                                    logger.error(f"Failed to NACK job {job_id}: {str(nack_err)}")
                        except Exception as e:
                            logger.error(f"Error processing message for job {job_id}: {str(e)}", exc_info=True)
                            try:
                                consume_connection.add_callback_threadsafe(
                                    lambda: ch.basic_nack(delivery_tag=delivery_tag, requeue=False)
                                )
                            except Exception as nack_err:
                                logger.error(f"Failed to NACK job {job_id} after error: {str(nack_err)}")

                    thread = threading.Thread(target=process, daemon=True, name=f"job-{job_id}")
                    thread.start()

                consume_channel.basic_consume(
                    queue=Config.ai_jobs,
                    on_message_callback=callback,
                    auto_ack=False
                )

                logger.info(f"Waiting for messages on queue: {Config.ai_jobs}")
                consume_channel.start_consuming()

            except (pika.exceptions.AMQPConnectionError, pika.exceptions.StreamLostError) as e:
                logger.error(f"Connection lost: {str(e)}. Reconnecting in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}. Reconnecting in 5 seconds...")
                time.sleep(5)
            finally:
                if consume_connection and not consume_connection.is_closed:
                    try:
                        consume_connection.close()
                    except Exception:
                        pass


rabbitmq_client = RabbitMQClient()
