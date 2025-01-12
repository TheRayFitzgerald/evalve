from typing import Dict, Any, List
from agents import ClassificationAgent
from config import EvalItem

def evaluate_agent(agent: ClassificationAgent, eval_dataset: List[EvalItem]) -> Dict[str, Any]:
    results = []
    correct = 0
    total = len(eval_dataset)

    for item in eval_dataset:
        prediction = agent.classify(item["ticket"])
        is_correct = prediction == item["ground_truth"]
        if is_correct:
            correct += 1

        results.append({
            "ticket": item["ticket"],
            "predicted": prediction,
            "actual": item["ground_truth"],
            "correct": is_correct
        })

    accuracy = correct / total
    return {
        "accuracy": accuracy,
        "detailed_results": results
    } 