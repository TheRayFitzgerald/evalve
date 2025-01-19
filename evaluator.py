from typing import cast
from agents import ClassificationAgent
from agents.improvement_agent import EvaluationResults, TicketResult
from config import EvalItem

def evaluate_agent(agent: ClassificationAgent, eval_dataset: list[EvalItem]) -> EvaluationResults:
    results: list[TicketResult] = []
    correct = 0
    total = len(eval_dataset)

    for item in eval_dataset:
        prediction = agent.classify(item["ticket"])
        is_correct = prediction == item["ground_truth"]
        if is_correct:
            correct += 1

        # Cast prediction to Literal type since we know it's one of the valid values
        results.append(cast(TicketResult, {
            "ticket": item["ticket"],
            "predicted": prediction,
            "actual": item["ground_truth"],
            "correct": is_correct
        }))

    accuracy = correct / total
    return {
        "accuracy": accuracy,
        "detailed_results": results
    } 