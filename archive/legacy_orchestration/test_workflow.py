import pytest

# Dummy orchestrator for illustration; replace with actual imports
class Workflow:
    def run(self, data):
        return {'status': 'ok', 'processed': True}

def test_workflow_run():
    wf = Workflow()
    result = wf.run({'foo': 'bar'})
    assert result['status'] == 'ok'
    assert result['processed'] is True
