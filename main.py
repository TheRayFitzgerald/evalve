from config import ANTHROPIC_API_KEY, INITIAL_PROMPT, EVAL_DATASET
from agent import ClassificationAgent
from evaluator import evaluate_agent

def main():
    # Initialize agent with initial prompt
    current_prompt = INITIAL_PROMPT
    agent = ClassificationAgent(ANTHROPIC_API_KEY, current_prompt)
    
    max_iterations = 5
    best_accuracy = 0
    best_prompt = current_prompt
    
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}")
        print("-" * 50)
        
        # Evaluate current performance
        eval_results = evaluate_agent(agent, EVAL_DATASET)
        current_accuracy = eval_results["accuracy"]
        print(f"Current accuracy: {current_accuracy:.2%}")
        
        # Keep track of best performance
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_prompt = current_prompt
        
        # Try to improve prompt
        current_prompt = agent.improve_prompt(eval_results, current_prompt)
        agent = ClassificationAgent(ANTHROPIC_API_KEY, current_prompt)
        
        # Break if perfect accuracy is achieved
        if current_accuracy == 1.0:
            print("Perfect accuracy achieved!")
            break
    
    print("\nFinal Results:")
    print(f"Best accuracy: {best_accuracy:.2%}")
    print(f"Best prompt:\n{best_prompt}")

if __name__ == "__main__":
    main() 