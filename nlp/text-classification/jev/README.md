# Jev

A quick tour of [Jev](https://docs.typesafe.ai), TypeSafe's first *System One* model, using a few handmade examples: how it works, whether it's any good, how fast and how cheap it is, and what it is (and isn't) good for.

Jev answers typed questions about a piece of state and returns probabilities instead of text: **Noul** (yes/no → probability of yes), **Choice** (one option from a set) and **Score** (ordered levels).

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- An [OpenRouter](https://openrouter.ai) API key

### Installation

```bash
uv sync
echo "OPENROUTER_API_KEY=sk-or-..." > .env   # git-ignored
uv run python -m ipykernel install --user --name=jev --display-name "Python (jev)"
```

### Calling Jev through OpenRouter

Jev is a decisions model, not a chat model: OpenRouter rejects it on `/chat/completions`. Use TypeSafe's SDK pointed at OpenRouter:

```python
from typesafe_sdk import Noul, TypeSafeClient

client = TypeSafeClient(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api", model="typesafe/jev-1.13")
response = client.system_one("Not bad at all, I'd happily buy it again.", {"satisfied": Noul(instructions="The customer is satisfied.")})
print(response.nouls["satisfied"].noul)  # probability of yes
```

## Notebook

`jev.ipynb` runs one example of each question type, then a few handmade cases per type (including sarcasm, negation and ambiguous tickets), two of TypeSafe's documented weak spots, latency with 1 to 20 questions per call, and the cost of every call it made.

## Resources

- [TypeSafe documentation](https://docs.typesafe.ai)
- [Jev 1.13 known weak spots](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)
- [TypeSafe Python SDK](https://docs.typesafe.ai/sdk/python.md)
