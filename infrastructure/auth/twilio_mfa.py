class TwilioMFA:
    def send_mfa_code(self, phone_number):
        return "mock_verification_sid"

    def verify_mfa_code(self, phone_number, code):
        return True
