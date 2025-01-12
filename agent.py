from anthropic import Anthropic
from typing import List, Dict, Any

class ClassificationAgent:
    def __init__(self, api_key: str, system_prompt: str):
        self.client = Anthropic(api_key=api_key)
        self.system_prompt = system_prompt

    def classify(self, ticket: str) -> str:
        messages = [
            {
                "role": "user",
                "content": f"Ticket: {ticket}\nClassification:"
            }
        ]
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=messages,
            system=self.system_prompt
        )
        
        return response.content[0].text.strip().lower()

    def improve_prompt(self, eval_results: Dict[str, Any], current_prompt: str) -> str:
        # Convert eval_results to a formatted string for better readability
        results_str = "\n".join([
            f"Ticket: {result['ticket']}\n"
            f"Predicted: {result['predicted']}\n"
            f"Actual: {result['actual']}\n"
            f"Correct: {result['correct']}\n"
            for result in eval_results['detailed_results']
        ])
        
        improvement_prompt = f"""You are a self-improving AI agent. Here are the evaluation results of your performance as a ticket classification agent:

Current system prompt:
{current_prompt}

Current accuracy: {eval_results['accuracy']:.2%}

Evaluation results:
{results_str}

Please analyze these results and suggest an improved system prompt that would lead to better classification accuracy. Respond with only the new prompt text."""

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2048,
            messages=[{"role": "user", "content": improvement_prompt}]
        )
        
        return response.content[0].text.strip() 