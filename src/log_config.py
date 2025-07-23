import logging.config
import os

import yaml

LOGGING_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../Golden/config/logging.yaml')
)

def setup_logging():
    if os.path.exists(LOGGING_CONFIG_PATH):
        with open(LOGGING_CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
