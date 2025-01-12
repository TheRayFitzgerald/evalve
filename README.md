# Evalve 🤖

A proof-of-concept for a self-improving AI agent that evolves through evaluation-driven prompt engineering.

## Overview

Evalve demonstrates autonomous agent improvement through iterative prompt refinement. The agent:

1. Starts with an initial prompt
2. Evaluates performance against ground truth data
3. Self-improves by analyzing evaluation results
4. Repeats until optimal performance or convergence

## Scientific Context

Evalve implements several key concepts from AI research:

- **Self-Improving Systems**: Demonstrates prompt-based self-improvement through performance feedback
- **Meta-Learning**: Implements prompt meta-learning where the system learns to optimize its own instructions
- **Recursive Self-Improvement**: Offers a controlled form of RSI focused on prompt optimization
- **Auto-Prompting**: Functions as a feedback-driven auto-prompting system
- **Self-Reflection**: Uses evaluation-driven self-reflection for improvement

Technically, Evalve is an "evaluation-based prompt self-improvement system" that explores controlled recursive optimization in language models.

## Current Implementation

The demo implements a support ticket urgency classifier (low/medium/high/critical) that improves its classification accuracy through self-reflection.

### Architecture

Evalve uses a multi-agent architecture:

- `ClassificationAgent`: The core agent that performs the classification task using Claude 3 Sonnet
- `ImprovementAgent`: A separate agent (Claude 3.5 Sonnet v2) that analyzes results and rewrites the core agent's prompt
- `evaluator.py`: Measures performance against ground truth
- `config.py`: Contains evaluation dataset and initial prompt

Using a simpler model for the core agent provides a clearer demonstration of how prompt engineering can improve performance, as there's more room for optimization through better instructions.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# Run the self-improvement loop
python main.py
```

## Features

- 🔄 Iterative self-improvement loop
- 📊 Performance evaluation against ground truth
- 🛑 Automatic stopping on regression or perfection
- 📝 Prompt engineering best practices
- 🔍 Detailed evaluation results

## Requirements

- Python 3.x
- Anthropic API key
- Dependencies listed in `requirements.txt`

## License

MIT 