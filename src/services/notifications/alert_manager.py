# Purpose: Manage alerts and notifications across the GoldenSignalsAI platform
# Provides mechanisms for sending alerts via various channels

import logging
from typing import Dict, Any, Optional

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages alerts and notifications for the trading system.
    Supports multiple notification channels and logging.
    """

    def __init__(self, channels: Optional[Dict[str, Any]] = None):
        """
        Initialize the AlertManager with optional notification channels.

        Args:
            channels (Dict[str, Any], optional): Configuration for notification channels.
        """
        self.channels = channels or {}
        logger.info({"message": "AlertManager initialized", "channels": list(self.channels.keys())})

    def send_trading_alert(self, trading_decision: Dict[str, Any]):
        """
        Send an alert about a trading decision.

        Args:
            trading_decision (Dict[str, Any]): Details of the trading decision.
        """
        import os
        import json
        from .notify_channels import send_slack, send_discord, send_email, send_push, send_sms
        
        try:
            logger.info({
                "message": "Trading Decision Alert",
                "action": trading_decision.get('action', 'N/A'),
                "confidence": trading_decision.get('confidence', 0.0),
                "metadata": trading_decision.get('metadata', {})
            })
            username = trading_decision.get('username', 'user1')
            NOTIFICATION_FILE = os.getenv('NOTIFICATION_SETTINGS_FILE', 'user_notification_settings.json')
            if os.path.exists(NOTIFICATION_FILE):
                with open(NOTIFICATION_FILE, 'r') as f:
                    user_settings = json.load(f).get(username, {})
            else:
                user_settings = {}
            # Only notify if confidence is high enough
            high_conf = user_settings.get('highConfidenceOnly', True)
            if high_conf and trading_decision.get('confidence', 0.0) < 0.8:
                logger.info({"message": "Not sending alert: confidence below threshold", "confidence": trading_decision.get('confidence', 0.0)})
                return
            msg = f"Trade Alert: {trading_decision.get('action')} @ {trading_decision.get('price', 'N/A')} | Confidence: {trading_decision.get('confidence', 0.0):.2f}"
            # Slack
            slack_url = user_settings.get('slack')
            if slack_url:
                ok = send_slack(slack_url, msg)
                logger.info({"message": "Slack notification sent", "ok": ok})
            # Discord
            discord_url = user_settings.get('discord')
            if discord_url:
                ok = send_discord(discord_url, msg)
                logger.info({"message": "Discord notification sent", "ok": ok})
            # Email
            email_addr = user_settings.get('email')
            if email_addr:
                ok = send_email(email_addr, "GoldenSignalsAI Trade Alert", msg)
                logger.info({"message": "Email notification sent", "ok": ok})
            # SMS
            sms_number = user_settings.get('sms')
            if sms_number:
                ok = send_sms(sms_number, msg)
                logger.info({"message": "SMS notification sent", "ok": ok})
            # Push
            if user_settings.get('push'):
                device_token = user_settings.get('device_token', None)
                if device_token:
                    ok = send_push(device_token, msg)
                    logger.info({"message": "Push notification sent", "ok": ok})
        except Exception as e:
            logger.error({"message": f"Failed to send trading alert: {str(e)}"})

    def send_error_alert(self, error_message: str):
        """
        Send an alert about a system error.

        Args:
            error_message (str): Description of the error.
        """
        try:
            # Log the error
            logger.error({
                "message": "System Error Alert",
                "error_details": error_message
            })

            # Placeholder for additional error notification channels
            # In a real implementation, this would send alerts via email, SMS, etc.
        except Exception as e:
            logger.error({"message": f"Failed to send error alert: {str(e)}"})

    def send_performance_alert(self, performance_metrics: Dict[str, Any]):
        """
        Send an alert about system performance.

        Args:
            performance_metrics (Dict[str, Any]): Performance-related metrics.
        """
        try:
            # Log performance metrics
            logger.info({
                "message": "Performance Alert",
                "metrics": performance_metrics
            })

            # Placeholder for additional performance notification channels
            # In a real implementation, this would send alerts via email, dashboard, etc.
        except Exception as e:
            logger.error({"message": f"Failed to send performance alert: {str(e)}"})
