import logging

from GoldenSignalsAI.application.services.alert_factory import AlertFactory
from GoldenSignalsAI.application.services.alert_service import AlertService
from GoldenSignalsAI.domain.trading.entities.user_preferences import UserPreferences

logger = logging.getLogger(__name__)

class EventHandler:
    def __init__(self):
        self.alert_service = AlertFactory.create_alert_service()

    async def handle_price_alert(self, event):
        logger.info(f"Handling PriceAlertEvent for {event.symbol}")
        user_prefs = UserPreferences(user_id=1, phone_number="+1234567890", whatsapp_number="+1234567890",
                                     x_enabled=True, enabled_channels=["sms", "whatsapp", "x"],
                                     price_threshold=event.threshold)
        await self.alert_service.send_alert(user_prefs, event)

    async def handle_signal_event(self, event):
        logger.info(f"Handling SignalEvent for {event.symbol}")
        user_prefs = UserPreferences(user_id=1, phone_number="+1234567890", whatsapp_number="+1234567890",
                                     x_enabled=True, enabled_channels=["sms", "whatsapp", "x"],
                                     price_threshold=0)
        await self.alert_service.send_alert(user_prefs, event)
