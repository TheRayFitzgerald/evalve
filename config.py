import os
from dotenv import load_dotenv
from typing import TypedDict, List
from inspect import cleandoc

load_dotenv()

# Ensure API key is available
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Add type annotation to ensure it's treated as str
ANTHROPIC_API_KEY: str = ANTHROPIC_API_KEY


class EvalItem(TypedDict):
    ticket: str
    ground_truth: str


INITIAL_PROMPT = cleandoc(
    """
    You are a customer support ticket classification agent. Your task is to classify tickets into one of these urgency levels: low, medium, high, critical.

    Classification criteria:
    - Low: General inquiries, feature requests, documentation issues
    - Medium: Minor bugs, account issues, billing questions
    - High: Service disruptions, data access issues, security concerns
    - Critical: System outages, data breaches, severe security incidents

    Respond only with the classification label in lowercase.
    """
)

EVAL_DATASET: List[EvalItem] = [
    {
        "ticket": "I can't find where to change my password",
        "ground_truth": "low",
    },
    {
        "ticket": "Our production database is down, nothing is working!",
        "ground_truth": "critical",
    },
    {
        "ticket": "Getting 404 errors on the login page",
        "ground_truth": "high",
    },
]
