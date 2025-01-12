import os
from dotenv import load_dotenv
from typing import TypedDict, List

load_dotenv()

# Ensure API key is available
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

class EvalItem(TypedDict):
    ticket: str
    ground_truth: str

EVAL_DATASET: List[EvalItem] = [
    {
        "ticket": "I can't find where to change my password",
        "ground_truth": "low"
    },
    {
        "ticket": "Our production database is down, nothing is working!",
        "ground_truth": "critical"
    },
    {
        "ticket": "Getting 404 errors on the login page",
        "ground_truth": "high"
    },
]

# Initial system prompt for the classification agent
INITIAL_PROMPT = """You are a customer support ticket classification agent. Your task is to classify tickets into one of these urgency levels: low, medium, high, critical.

Classification criteria:
- Low: General inquiries, feature requests, documentation issues
- Medium: Minor bugs, account issues, billing questions
- High: Service disruptions, data access issues, security concerns
- Critical: System outages, data breaches, severe security incidents

Respond only with the classification label in lowercase."""

# Sample evaluation dataset
EVAL_DATASET = [
    {
        "ticket": "I can't find where to change my password",
        "ground_truth": "low"
    },
    {
        "ticket": "Our production database is down, nothing is working!",
        "ground_truth": "critical"
    },
    {
        "ticket": "Getting 404 errors on the login page",
        "ground_truth": "high"
    },
    # Add more examples as needed
] 