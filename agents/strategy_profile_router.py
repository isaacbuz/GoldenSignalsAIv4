import json
import os


class StrategyProfileRouter:
    def __init__(self, config_path="strategy_profiles.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def get_profile(self, profile_key):
        return self.config.get(profile_key, self.config["swing"])  # default fallback
