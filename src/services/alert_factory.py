from src.infrastructure.external_services.twilio_sms import TwilioSMSClient
from src.infrastructure.external_services.twilio_whatsapp import TwilioWhatsAppClient
from src.infrastructure.external_services.x_api import XClient
from src.services.alert_service import AlertService


class AlertFactory:
    @staticmethod
    def create_alert_service():
        return AlertService(TwilioSMSClient(), TwilioWhatsAppClient(), XClient())
