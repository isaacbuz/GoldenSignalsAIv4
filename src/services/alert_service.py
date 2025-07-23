from GoldenSignalsAI.domain.trading.entities.user_preferences import UserPreferences
from GoldenSignalsAI.infrastructure.external_services.twilio_sms import TwilioSMSClient
from GoldenSignalsAI.infrastructure.external_services.twilio_whatsapp import TwilioWhatsAppClient
from GoldenSignalsAI.infrastructure.external_services.x_api import XClient


class AlertService:
    def __init__(self, twilio_sms_client, twilio_whatsapp_client, x_client):
        self.twilio_sms_client = twilio_sms_client
        self.twilio_whatsapp_client = twilio_whatsapp_client
        self.x_client = x_client
        self.alert_channels = {"sms": self.twilio_sms_client, "whatsapp": self.twilio_whatsapp_client, "x": self.x_client}

    async def send_alert(self, user_preferences: UserPreferences, event):
        message = self._format_message(event)
        for channel in user_preferences.enabled_channels:
            if channel in self.alert_channels:
                self.alert_channels[channel].send(message, user_preferences)

    def _format_message(self, event):
        if event["type"] == "PriceAlertEvent":
            return f"{event['symbol']} price exceeded {event['threshold']}! Current price: {event['price']}"
        elif event["type"] == "SignalEvent":
            return f"GoldenSignalsAI Alert: {event['action']} signal for {event['symbol']} at {event['price']}. #TradingSignals"
        return "Unknown event type"
