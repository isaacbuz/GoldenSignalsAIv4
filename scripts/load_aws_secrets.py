"""
Load secrets from AWS Secrets Manager and inject them as environment variables.
Call this script at the very top of your main entrypoint (before any imports that use env vars).
"""
import boto3
import os
import json

def load_aws_secret(secret_name, region_name="us-east-1"):
    client = boto3.client("secretsmanager", region_name=region_name)
    get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    secret = get_secret_value_response["SecretString"]
    return json.loads(secret)

def inject_env_vars_from_secret(secret_name, region_name="us-east-1"):
    secrets = load_aws_secret(secret_name, region_name)
    for k, v in secrets.items():
        os.environ[k] = v

if __name__ == "__main__":
    secret_name = os.getenv("AWS_SECRET_ENV_NAME", "prod/goldensignalsai/env")
    region_name = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    inject_env_vars_from_secret(secret_name, region_name)
    print(f"AWS Secrets from '{secret_name}' loaded and environment variables set.")
