import os

from notifications.alert_manager import AlertManager

# Prepare a fake user config for testing
test_user = "testuser"
settings = {
    test_user: {
        "slack": os.getenv("TEST_SLACK_WEBHOOK", ""),
        "discord": os.getenv("TEST_DISCORD_WEBHOOK", ""),
        "email": os.getenv("TEST_EMAIL", ""),
        "sms": os.getenv("TEST_SMS", ""),
        "push": False,
        "highConfidenceOnly": False
    }
}

with open("user_notification_settings.json", "w") as f:
    import json
    json.dump(settings, f, indent=2)

# Simulate a trading decision
alert_mgr = AlertManager()
trading_decision = {
    "username": test_user,
    "action": "BUY",
    "price": 123.45,
    "confidence": 0.95,
    "metadata": {"symbol": "AAPL"}
}

alert_mgr.send_trading_alert(trading_decision)
print("Notification delivery test completed. Check logs and your channels.")
