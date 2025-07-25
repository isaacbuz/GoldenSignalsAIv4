import json
import os

import boto3

SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")
sqs = boto3.client("sqs")

def enqueue_retrain_job(agent: str):
    sqs.send_message(
        QueueUrl=SQS_QUEUE_URL,
        MessageBody=json.dumps({"agent": agent})
    )
