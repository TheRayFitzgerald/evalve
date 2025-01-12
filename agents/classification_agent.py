from .base import BaseAgent
from anthropic.types import MessageParam


class ClassificationAgent(BaseAgent):
    def classify(self, ticket: str) -> str:
        messages: list[MessageParam] = [
            {
                "role": "user",
                "content": f"Ticket: {ticket}\nClassification:",
            }
        ]

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=messages,
            system=self.system_prompt,
        )

        return response.content[0].text.strip().lower()
