import boto3
import os
import json
import time

sqs = boto3.client("sqs")
QUEUE_URL = os.getenv("SQS_QUEUE_URL")

while True:
    resp = sqs.receive_message(QueueUrl=QUEUE_URL, MaxNumberOfMessages=1, WaitTimeSeconds=20)
    for msg in resp.get("Messages", []):
        job = json.loads(msg["Body"])
        agent = job["agent"]
        # TODO: Call your retrain logic here (e.g., subprocess or SageMaker)
        print(f"Retraining agent: {agent}")
        sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=msg["ReceiptHandle"])
    time.sleep(2)
