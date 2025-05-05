import os
import json
from kafka import KafkaProducer

class EventPublisher:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BROKER", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    async def publish(self, topic, event):
        self.producer.send(topic, event)
        self.producer.flush()

    def close(self):
        self.producer.close()
