class PipelineAgent:
    def __init__(self, name):
        self.name = name

    def run(self, input_data, context):
        raise NotImplementedError("Each pipeline agent must define a run method.")
