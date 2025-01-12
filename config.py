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
    {
        "ticket": "System is running slow",
        "ground_truth": "medium",
    },
    {
        "ticket": "Hey, just wondering if you could take a look at this weird charge on my account? Someone might have stolen my credit card info",
        "ground_truth": "high",
    },
    {
        "ticket": "URGENT!!! Need help ASAP! Can't change my profile picture!!!",
        "ground_truth": "low",
    },
    {
        "ticket": "The app is slow and I found a typo in the docs. Also having trouble logging in sometimes.",
        "ground_truth": "high",
    },
    {
        "ticket": "Observed 500ms p99 latency spike in EU region, 15% error rate increase in last 5m",
        "ground_truth": "high",
    },
    {
        "ticket": "FYI - noticed my coworker's salary info is visible on my dashboard",
        "ground_truth": "critical",
    },
    {
        "ticket": "Need to update pricing before tomorrow's launch",
        "ground_truth": "high",
    },
    {
        "ticket": "Some customer health data is showing up unencrypted in the logs",
        "ground_truth": "critical",
    },
    {
        "ticket": "The system is broken - it doesn't let me upload videos larger than 1GB",
        "ground_truth": "low",
    },
    {
        "ticket": "Why does the URL show my account number? My friend said that's bad.",
        "ground_truth": "high",
    },
    {
        "ticket": "Payment processing is delayed by 24 hours",
        "ground_truth": "critical",
    },
    {
        "ticket": "Something weird happening with user accounts",
        "ground_truth": "high",
    },
]
