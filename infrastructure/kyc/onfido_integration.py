from onfido import Onfido


class KYCManager:
    def __init__(self):
        self.onfido = Onfido(api_token=os.getenv('ONFIDO_API_TOKEN'))

    async def verify_user(self, user_id):
        applicant = self.onfido.Applicant.create({
            'first_name': user_id.first_name,
            'last_name': user_id.last_name
        })
        return self.onfido.Check.create(applicant.id, {
            'type': 'standard',
            'reports': ['identity', 'document', 'facial_similarity']
        })
