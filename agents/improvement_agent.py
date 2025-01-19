from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from inspect import cleandoc
from .base import BaseAgent

@dataclass
class PromptChange:
    timestamp: datetime
    old_prompt: str
    new_prompt: str
    motivation: str
    accuracy_before: float
    accuracy_after: Optional[float] = None
    fixed_cases: List[str] = field(default_factory=list)
    regression_cases: List[str] = field(default_factory=list)

class PromptHistory:
    def __init__(self):
        self.changes: List[PromptChange] = []
    
    def add_change(self, change: PromptChange):
        self.changes.append(change)
    
    def update_last_change_results(self, accuracy: float, fixed_cases: List[str], regression_cases: List[str]):
        if self.changes:
            self.changes[-1].accuracy_after = accuracy
            self.changes[-1].fixed_cases = fixed_cases
            self.changes[-1].regression_cases = regression_cases

class ImprovementAgent(BaseAgent):
    def __init__(self, api_key: str, system_prompt: str):
        super().__init__(api_key, system_prompt)
        self.history = PromptHistory()

    def _analyze_changes(self, eval_results: Dict[str, Any], previous_results: Optional[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        fixed_cases = []
        regression_cases = []
        
        if previous_results:
            prev_results_map = {r["ticket"]: r for r in previous_results["detailed_results"]}
            current_results_map = {r["ticket"]: r for r in eval_results["detailed_results"]}
            
            # Find fixed and regression cases
            for ticket, curr_result in current_results_map.items():
                if ticket in prev_results_map:
                    prev_correct = prev_results_map[ticket]["correct"]
                    curr_correct = curr_result["correct"]
                    
                    if not prev_correct and curr_correct:
                        fixed_cases.append(ticket)
                    elif prev_correct and not curr_correct:
                        regression_cases.append(ticket)
        
        return fixed_cases, regression_cases

    def improve_prompt(self, eval_results: Dict[str, Any], current_prompt: str, previous_results: Optional[Dict[str, Any]] = None) -> str:
        fixed_cases, regression_cases = self._analyze_changes(eval_results, previous_results)
        
        # Create history summary
        history_summary = ""
        if self.history.changes:
            last_change = self.history.changes[-1]
            accuracy_str = f"{last_change.accuracy_after:.2%}" if last_change.accuracy_after is not None else "Not evaluated yet"
            history_summary = cleandoc(f"""
                Previous change summary:
                - Motivation: {last_change.motivation}
                - Previous accuracy: {last_change.accuracy_before:.2%}
                - New accuracy: {accuracy_str}
                - Fixed cases: {len(last_change.fixed_cases)}
                - Regression cases: {len(last_change.regression_cases)}
            """)

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

            {history_summary if history_summary else ""}

            Evaluation results:
            {results_str}

            Fixed cases from last change: {fixed_cases}
            Regression cases from last change: {regression_cases}
            </task>

            <guidelines>
            Follow prompt engineering best practices:
            1. Be clear and direct in instructions
            2. Use examples (multishot prompting) when helpful
            3. Enable chain-of-thought reasoning
            4. Use XML tags to structure different components
            5. Keep the prompt focused and specific
            6. Consider the impact of previous changes
            7. Avoid changes that might cause regressions
            </guidelines>

            <output_format>
            First, provide a one-line motivation for your changes starting with "MOTIVATION: ".
            Then, on a new line, provide the new improved prompt text.
            </output_format>
            """
        )

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": improvement_prompt}],
        )

        response_text = response.content[0].text.strip()
        
        # Parse motivation and new prompt
        motivation_line, new_prompt = response_text.split("\n", 1)
        motivation = motivation_line.replace("MOTIVATION: ", "").strip()
        new_prompt = new_prompt.strip()

        # Record the change
        change = PromptChange(
            timestamp=datetime.now(),
            old_prompt=current_prompt,
            new_prompt=new_prompt,
            motivation=motivation,
            accuracy_before=eval_results["accuracy"]
        )
        self.history.add_change(change)

        return new_prompt 