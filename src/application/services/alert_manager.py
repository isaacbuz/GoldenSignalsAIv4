import smtplib
import requests
from email.message import EmailMessage
from typing import List

class AlertManager:
    def __init__(self, email_config: dict = None, slack_webhook: str = "", telegram_config: dict = None):
        self.email_config = email_config or {}
        self.slack_webhook = slack_webhook
        self.telegram_config = telegram_config or {}

    def send_email(self, to_emails: List[str], subject: str, body: str):
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = self.email_config.get("sender")
        msg['To'] = ", ".join(to_emails)

        with smtplib.SMTP_SSL(self.email_config.get("smtp_server"), self.email_config.get("port", 465)) as server:
            server.login(self.email_config.get("sender"), self.email_config.get("password"))
            server.send_message(msg)

    def send_slack_alert(self, message: str):
        if self.slack_webhook:
            requests.post(self.slack_webhook, json={"text": message})

    def send_telegram_alert(self, message: str):
        token = self.telegram_config.get("bot_token")
        chat_id = self.telegram_config.get("chat_id")
        if token and chat_id:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, data={"chat_id": chat_id, "text": message})

    def alert_all(self, subject: str, message: str, recipients: List[str]):
        self.send_email(recipients, subject, message)
        self.send_slack_alert(message)
        self.send_telegram_alert(message)
