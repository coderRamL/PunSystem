# 🎭 PunSystem Agent

An intelligent NLP agent built with the **Strands SDK** that analyzes puns by identifying ambiguous nouns and explaining their meanings using **spaCy** and **NLTK WordNet**.

---

## 🛠 Prerequisites

* **Python 3.12**: This project requires Python 3.12 to ensure compatibility with the `pydantic.v1` schemas used by spaCy.
* **uv**: Python package manager used for environment and dependency management.
* **Ollama Cloud**: An account and API key are required to power the agent's reasoning model.

---

## ⚙️ Configuration & Installation

### 1. Project Initialization
Clone the repository and set up the virtual environment using `uv`:
```bash
# Install and pin the correct Python version
uv python install 3.12
uv python pin 3.12

# Sync dependencies and create .venv
uv sync
```

### 2. Environment Variables
Create a .env file in the root directory to store your credentials:
```bash
OLLAMA_API_KEY=your_api_key_here
```

### 3. Linguistic Data Downloads
Linguistic models must be downloaded into the environment separately from the core libraries.

##### Download spaCy English Model:

*Note: Installing the wheel directly bypasses any issues with missing pip modules in some uv environments.*
```bash
uv pip install [https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl](https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl)
```
##### Download NLTK (WOrdNet):
```bash
uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 4. Project Structure
```Plaintext
PUNSYSTEM/
├── tools/
│   ├── __init__.py          # Required to treat directory as a package
│   └── ambiguous_words.py   # Main NLP logic (The Tool)
├── .env                     # Private API keys (Do not commit)
├── .gitignore               # Prevents tracking of __pycache__ and .venv
├── agent.py                 # Agent entry point and conversation loop
├── pyproject.toml           # Build and dependency configuration
└── README.md                # Project documentation
```

### 5. Usage
Launch the agent via terminal:
```bash
uv run agent.py
```
1. Enter a pun when prompted.
2. The agent will call `indentify_ambiguous_words` tool.
3. The agent will analyze the returned dictionary of definitions and explain the joke (or lack thereof) to you.

### Troubleshooting Applied

- Double Printing: `callback_handler=null_callback_handler`.
- Import Errors: __init__.py added to tools/ folder so python can identify tools.
- Pydantic ConfigError: Python version must be < 3.13 otherwise the script will not execute.
