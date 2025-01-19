from typing import Literal, TypedDict
from dataclasses import dataclass, field
from datetime import datetime
from inspect import cleandoc
from .base import BaseAgent

# Shared type for urgency levels
UrgencyLevel = Literal["low", "medium", "high", "critical"]

class TicketResult(TypedDict):
    ticket: str
    predicted: UrgencyLevel
    actual: UrgencyLevel
    correct: bool

class EvaluationResults(TypedDict):
    accuracy: float
    detailed_results: list[TicketResult]

@dataclass
class Case:
    ticket: str
    predicted: UrgencyLevel
    actual: UrgencyLevel
    description: str = ""

@dataclass
class PromptChange:
    timestamp: datetime
    old_prompt: str
    new_prompt: str
    motivation: str
    accuracy_before: float
    accuracy_after: float | None = None
    fixed_cases: list[Case] = field(default_factory=list)
    regression_cases: list[Case] = field(default_factory=list)

class PromptHistory:
    def __init__(self):
        self.changes: list[PromptChange] = []
    
    def add_change(self, change: PromptChange):
        self.changes.append(change)
    
    def update_last_change_results(self, accuracy: float, fixed_cases: list[Case], regression_cases: list[Case]):
        if self.changes:
            self.changes[-1].accuracy_after = accuracy
            self.changes[-1].fixed_cases = fixed_cases
            self.changes[-1].regression_cases = regression_cases

class ImprovementAgent(BaseAgent):
    def __init__(self, api_key: str, system_prompt: str):
        super().__init__(api_key, system_prompt)
        self.history = PromptHistory()

    def _analyze_changes(self, eval_results: EvaluationResults, previous_results: EvaluationResults | None) -> tuple[list[Case], list[Case]]:
        fixed_cases: list[Case] = []
        regression_cases: list[Case] = []
        
        if previous_results:
            prev_results_map = {r["ticket"]: r for r in previous_results["detailed_results"]}
            current_results_map = {r["ticket"]: r for r in eval_results["detailed_results"]}
            
            # Find fixed and regression cases
            for ticket, curr_result in current_results_map.items():
                if ticket in prev_results_map:
                    prev_result = prev_results_map[ticket]
                    
                    if not prev_result["correct"] and curr_result["correct"]:
                        fixed_cases.append(Case(
                            ticket=ticket,
                            predicted=curr_result["predicted"],
                            actual=curr_result["actual"],
                            description=f"Fixed: Was {prev_result['predicted']}, now correctly {curr_result['predicted']}"
                        ))
                    elif prev_result["correct"] and not curr_result["correct"]:
                        regression_cases.append(Case(
                            ticket=ticket,
                            predicted=curr_result["predicted"],
                            actual=curr_result["actual"],
                            description=f"Regressed: Was correct {prev_result['predicted']}, now wrong {curr_result['predicted']}"
                        ))
        
        return fixed_cases, regression_cases

    def improve_prompt(self, eval_results: EvaluationResults, current_prompt: str, previous_results: EvaluationResults | None = None) -> str:
        fixed_cases, regression_cases = self._analyze_changes(eval_results, previous_results)
        
        # Create history summary
        history_summary = ""
        if self.history.changes:
            last_change = self.history.changes[-1]
            accuracy_str = f"{last_change.accuracy_after:.2%}" if last_change.accuracy_after is not None else "Not evaluated yet"
            
            fixed_cases_str = "\n".join(f"  - {case.description}" for case in last_change.fixed_cases) if last_change.fixed_cases else "  None"
            regression_cases_str = "\n".join(f"  - {case.description}" for case in last_change.regression_cases) if last_change.regression_cases else "  None"
            
            history_summary = cleandoc(f"""
                Previous change summary:
                - Motivation: {last_change.motivation}
                - Previous accuracy: {last_change.accuracy_before:.2%}
                - New accuracy: {accuracy_str}
                - Fixed cases:
                {fixed_cases_str}
                - Regression cases:
                {regression_cases_str}
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

        current_fixed_str = "\n".join(f"  - {case.description}" for case in fixed_cases) if fixed_cases else "  None"
        current_regression_str = "\n".join(f"  - {case.description}" for case in regression_cases) if regression_cases else "  None"

        improvement_prompt = cleandoc(
            f"""<task>
            You are a prompt engineering expert. Your task is to improve the system prompt based on evaluation results.

            Current system prompt:
            {current_prompt}

            Current accuracy: {eval_results['accuracy']:.2%}

            {history_summary if history_summary else ""}

            Evaluation results:
            {results_str}

            Fixed cases from last change:
            {current_fixed_str}
            
            Regression cases from last change:
            {current_regression_str}
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