
import numpy as np


class Model:
    def __init__(self, config):
        self.provider = config["model_info"]["provider"]
        self.name = config["model_info"]["name"]
        self.temperature = float(config["params"]["temperature"])
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant. Your answer should be short and consice."},
            {"role": "user", "content": " "},
        ]        
    def print_model_info(self):
        print(f"{'-'*len(f'| Model name: {self.name}')}\n| Provider: {self.provider}\n| Model name: {self.name}\n{'-'*len(f'| Model name: {self.name}')}")

    def query(self, max_tokens=4096):
        pass

