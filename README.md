# 🔍 MetaFact

AI-powered political claim fact-checker using LLMs and live web search.

---

## Features

- **LLM-only mode** — instant verdict from the model's training knowledge
- **Search & verify mode** — the model performs up to 10 DuckDuckGo searches, cross-references sources, then delivers a verdict
- **Structured verdicts** — `TRUE · MOSTLY_TRUE · MISLEADING · MOSTLY_FALSE · FALSE · UNVERIFIABLE` with a confidence score
- **Web UI** via Streamlit and a **CLI** for scripting
- **OpenAI-compatible** — works with any LiteLLM-supported model or local endpoint

---

## Quickstart

### 1. Install dependencies

```bash
pip install uv        # if you don't have it
uv sync
```

### 2. Set your API key

```bash
export OPENAI_API_KEY=sk-...
```

### 3. Run

**Web UI**
```bash
streamlit run app.py
```

**CLI**
```bash
python cli.py "The US has the highest GDP per capita in the world."
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | API key (OpenAI or compatible provider) |
| `METAFACT_MODEL` | `openai/gpt-4o-mini` | LiteLLM model string |
| `OPENAI_API_BASE` | — | Custom base URL for OpenAI-compatible endpoints |

**Using a different model:**
```bash
METAFACT_MODEL=openai/gpt-4o python cli.py "..."
# or any LiteLLM-supported provider:
METAFACT_MODEL=anthropic/claude-3-5-sonnet-20241022 streamlit run app.py
```

**Using a local endpoint (e.g. Ollama, LM Studio):**
```bash
OPENAI_API_BASE=http://localhost:11434/v1 METAFACT_MODEL=openai/mistral streamlit run app.py
```

---

## Verdict scale

| Verdict | Meaning |
|---|---|
| `TRUE` | Accurate and well-supported |
| `MOSTLY_TRUE` | Largely accurate with minor caveats |
| `MISLEADING` | True facts used in a deceptive or out-of-context way |
| `MOSTLY_FALSE` | Grain of truth, but substantially inaccurate |
| `FALSE` | Factually incorrect |
| `UNVERIFIABLE` | Insufficient credible evidence |

---

## Requirements

- Python ≥ 3.11
- [`litellm`](https://github.com/BerriAI/litellm) ≥ 1.82
- [`ddgs`](https://github.com/deedy5/duckduckgo_search) ≥ 9.11
- [`streamlit`](https://streamlit.io) ≥ 1.40
