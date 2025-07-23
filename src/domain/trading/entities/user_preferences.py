from dataclasses import dataclass


@dataclass
class UserPreferences:
    user_id: int
    phone_number: str
    whatsapp_number: str
    x_enabled: bool
    enabled_channels: list
    price_threshold: float
