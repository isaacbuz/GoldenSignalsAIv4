import requests
import smtplib
from email.mime.text import MIMEText
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def send_slack(webhook_url: str, message: str) -> bool:
    try:
        resp = requests.post(webhook_url, json={"text": message})
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Slack notification failed: {e}")
        return False

def send_discord(webhook_url: str, message: str) -> bool:
    try:
        resp = requests.post(webhook_url, json={"content": message})
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Discord notification failed: {e}")
        return False

def send_email(to_email: str, subject: str, body: str, smtp_server: Optional[str]=None, smtp_user: Optional[str]=None, smtp_pass: Optional[str]=None) -> bool:
    try:
        smtp_server = smtp_server or 'localhost'
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = smtp_user or 'noreply@goldensignals.ai'
        msg['To'] = to_email
        with smtplib.SMTP(smtp_server) as server:
            if smtp_user and smtp_pass:
                server.starttls()
                server.login(smtp_user, smtp_pass)
            server.sendmail(msg['From'], [to_email], msg.as_string())
        return True
    except Exception as e:
        logger.error(f"Email notification failed: {e}")
        return False

# For push and sms, you would integrate with a provider (e.g., Firebase, Twilio)
def send_push(device_token: str, message: str) -> bool:
    logger.warning("Push notification not implemented.")
    return False

def send_sms(phone_number: str, message: str) -> bool:
    logger.warning("SMS notification not implemented.")
    return False
