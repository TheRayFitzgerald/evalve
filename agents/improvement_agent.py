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
            f"""<task>
            You are a prompt engineering expert. Your task is to improve the system prompt based on evaluation results.

            Current system prompt:
            {current_prompt}

            Current accuracy: {eval_results['accuracy']:.2%}

            Evaluation results:
            {results_str}
            </task>

            <guidelines>
            Follow prompt engineering best practices:
            1. Be clear and direct in instructions
            2. Use examples (multishot prompting) when helpful
            3. Enable chain-of-thought reasoning
            4. Use XML tags to structure different components
            5. Keep the prompt focused and specific
            </guidelines>

            <output_format>
            Respond ONLY with the new improved prompt text. Do not include any explanations or other text.
            The prompt should be self-contained and ready to use.
            </output_format>
            """
        )

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": improvement_prompt}],
        )

        return response.content[0].text.strip() 