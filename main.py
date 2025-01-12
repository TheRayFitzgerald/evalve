from config import ANTHROPIC_API_KEY, INITIAL_PROMPT, EVAL_DATASET
from agents import ClassificationAgent, ImprovementAgent
from evaluator import evaluate_agent

MAX_ITERATIONS = 10


def main():
    # Initialize agents
    current_prompt = INITIAL_PROMPT
    classification_agent = ClassificationAgent(ANTHROPIC_API_KEY, current_prompt)
    improvement_agent = ImprovementAgent(
        ANTHROPIC_API_KEY, ""
    )  # Empty prompt as it's not needed

    best_accuracy = 0
    best_prompt = current_prompt
    consecutive_regressions = 0

    for iteration in range(MAX_ITERATIONS):
        print(f"\nIteration {iteration + 1}")
        print("-" * 50)

        # Evaluate current performance
        eval_results = evaluate_agent(classification_agent, EVAL_DATASET)
        current_accuracy = eval_results["accuracy"]
        print(f"Current accuracy: {current_accuracy:.2%}")

        # Check for regression
        if current_accuracy < best_accuracy:
            consecutive_regressions += 1
            print(f"Regression detected ({consecutive_regressions} in a row)")
            if consecutive_regressions >= 2:
                print("Two consecutive regressions - stopping")
                break
        else:
            consecutive_regressions = 0
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_prompt = current_prompt

        # Try to improve prompt
        current_prompt = improvement_agent.improve_prompt(eval_results, current_prompt)
        classification_agent = ClassificationAgent(ANTHROPIC_API_KEY, current_prompt)

        # Break if perfect accuracy is achieved
        if current_accuracy == 1.0:
            print("Perfect accuracy achieved!")
            break

    print("\n" + "="*50)
    print("\033[92m")  # Set text color to green
    print("🎯 Final Results 🎯")
    print("="*50)
    print(f"Best accuracy: {best_accuracy:.2%}" + (" 🎊 🎉 🎊" if best_accuracy == 1.0 else ""))
    print("\nBest prompt:")
    print("-"*50)
    print(f"{best_prompt}")
    print("\033[0m")  # Reset text color
    print("="*50)


if __name__ == "__main__":
    main()
