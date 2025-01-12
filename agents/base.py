from anthropic import Anthropic
from typing import Dict, Any

class BaseAgent:
    def __init__(self, api_key: str, system_prompt: str):
        self.client = Anthropic(api_key=api_key)
        self.system_prompt = system_prompt 