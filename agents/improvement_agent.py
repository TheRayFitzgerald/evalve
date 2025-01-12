from typing import Dict, Any
from inspect import cleandoc
from .base import BaseAgent

class ImprovementAgent(BaseAgent):
    def improve_prompt(self, eval_results: Dict[str, Any], current_prompt: str) -> str:
        results_str = "\n".join(
            [
                f"Ticket: {result['ticket']}\n"
                f"Predicted: {result['predicted']}\n"
                f"Actual: {result['actual']}\n"
                f"Correct: {result['correct']}\n"
                for result in eval_results["detailed_results"]
            ]
        )

        improvement_prompt = cleandoc(
            f"""
            You are a self-improving AI agent. Here are the evaluation results of your performance as a ticket classification agent:

            Current system prompt:
            {current_prompt}

            Current accuracy: {eval_results['accuracy']:.2%}

            Evaluation results:
            {results_str}

            Please analyze these results and suggest an improved system prompt that would lead to better classification accuracy. Respond with only the new prompt text.
            """
        )

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": improvement_prompt}],
        )

        return response.content[0].text.strip() 