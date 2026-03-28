"""
Fact-checker core logic for political claims using LiteLLM and web search.

The CLI entry point lives in cli.py.
"""

import json
import os
from typing import Any

import litellm
from ddgs import DDGS

# ── Configuration from environment variables ──────────────────────────────────
DEFAULT_MODEL = os.getenv("METAFACT_MODEL", "openai/gpt-4o-mini")

_api_base = os.getenv("OPENAI_API_BASE", "")
if _api_base:
    litellm.api_base = _api_base

MAX_TOOL_CALLS = 10


def _format_results(results: list[dict]) -> str:
    """Format DuckDuckGo search results into a readable string for the LLM."""
    if not results:
        return "No results found."
    formatted = []
    for r in results:
        title = r.get("title", "")
        url = r.get("href", r.get("url", ""))
        body = r.get("body", "")
        formatted.append(f"Title: {title}\nURL: {url}\nSnippet: {body}")
    return "\n\n---\n\n".join(formatted)


def _parse_json_response(content: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from an LLM response string."""
    start = content.find("{")
    end = content.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass
    return {
        "verdict": "UNVERIFIABLE",
        "confidence": 0.0,
        "summary": content,
        "justification": content,
        "sources": [],
    }


def search_web(query: str, max_results: int = 6) -> str:
    """Unrestricted DuckDuckGo web search."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return _format_results(results)
    except Exception as e:
        return f"Search error: {e}"


_VERDICT_SCHEMA = """\
Respond ONLY with a JSON object (no markdown fences):
{
  "verdict": "<TRUE|MOSTLY_TRUE|MISLEADING|MOSTLY_FALSE|FALSE|UNVERIFIABLE>",
  "confidence": <0.0–1.0>,
  "summary": "<2–3 sentence summary of findings>",
  "justification": "<detailed explanation>",
  "sources": ["<URL or source name>", ...]
}

Verdict definitions:
- TRUE: Claim is accurate and well-supported.
- MOSTLY_TRUE: Claim is largely accurate but omits context or has minor inaccuracies.
- MISLEADING: Claim uses true facts in a deceptive or out-of-context way.
- MOSTLY_FALSE: Claim has a grain of truth but is substantially inaccurate.
- FALSE: Claim is factually incorrect.
- UNVERIFIABLE: Insufficient credible evidence to determine truth or falsehood."""

MODEL_ONLY_SYSTEM_PROMPT = f"""\
You are a rigorous, non-partisan fact-checker specializing in political claims.
You must rely solely on your training knowledge — no external search is available.

Your task:
1. Reason carefully about the claim based on your knowledge.
2. Distinguish between confirmed facts, contested claims, and outright falsehoods.
3. Be honest about the limits of your knowledge and training cutoff.

{_VERDICT_SCHEMA}
"""

SEARCH_SYSTEM_PROMPT = f"""\
You are a rigorous, non-partisan fact-checker specializing in political claims.

Your task:
1. Search the web up to {MAX_TOOL_CALLS} times to gather evidence.
2. Use `search_web` to find information relevant to the claim.
3. Cross-reference multiple sources before reaching a verdict.
4. Reason carefully about the evidence — distinguish between confirmed facts,
   contested claims, and outright falsehoods.

{_VERDICT_SCHEMA}
"""

SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information relevant to the claim.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find evidence for or against the claim.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


def verify_claim_model(claim: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """
    Verify a political claim using only the language model's training knowledge.

    No web search tools are used. The model reasons solely from its internal
    knowledge and is honest about uncertainty or knowledge cutoff limits.

    Args:
        claim: The political fact claim to verify.
        model: LiteLLM model string (e.g. "openai/gpt-4o-mini").

    Returns:
        Dict with keys: verdict, confidence, summary, justification, sources.
    """
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": MODEL_ONLY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Please fact-check this claim:\n\n\"{claim}\""},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    return _parse_json_response(content)


def verify_claim_search(claim: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """
    Verify a political claim using an LLM with unrestricted web search.

    The LLM searches the open web directly via `search_web` and produces a
    structured JSON verdict after gathering sufficient evidence.

    Args:
        claim: The political fact claim to verify.
        model: LiteLLM model string (e.g. "openai/gpt-4o-mini").

    Returns:
        Dict with keys: verdict, confidence, summary, justification, sources.
    """
    messages: list[dict] = [
        {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
        {"role": "user", "content": f"Please fact-check this claim:\n\n\"{claim}\""},
    ]

    tool_calls_made = 0

    while True:
        tool_choice = "none" if tool_calls_made >= MAX_TOOL_CALLS else "auto"

        response = litellm.completion(
            model=model,
            messages=messages,
            tools=SEARCH_TOOLS,
            tool_choice=tool_choice,
        )

        assistant_msg = response.choices[0].message
        messages.append(assistant_msg)

        if assistant_msg.tool_calls:
            for tc in assistant_msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                print(f"  [tool] search_web: {args.get('query', '')!r}")
                result = search_web(args.get("query", ""))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": result,
                    }
                )
            tool_calls_made += len(assistant_msg.tool_calls)
        else:
            content = (assistant_msg.content or "").strip()
            return _parse_json_response(content)


