class PipelineRunner:
    def __init__(self, agents):
        self.agents = agents  # List of PipelineAgent instances

    def run(self, input_data, context):
        data = input_data
        for agent in self.agents:
            try:
                data = agent.run(data, context)
            except Exception as e:
                data['error'] = str(e)
                break
        return data
