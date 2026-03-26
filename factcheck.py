"""
Fact-checker core logic for political claims using LiteLLM and web search.

The CLI entry point lives in cli.py.
"""

import json
import os
from typing import Any

import litellm
from duckduckgo_search import DDGS

# ── Configuration from environment variables ──────────────────────────────────
DEFAULT_MODEL = os.getenv("METAFACT_MODEL", "openai/gpt-4o-mini")

_api_key = os.getenv("OPENAI_API_KEY", "")
_api_base = os.getenv("OPENAI_API_BASE", "")
if _api_key:
    os.environ["OPENAI_API_KEY"] = _api_key
if _api_base:
    litellm.api_base = _api_base

# Reputable fact-checking and news sources to prefer in the first search pass
REPUTABLE_SITES = [
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "factcheck.org",
    "politifact.com",
    "snopes.com",
    "theguardian.com",
    "economist.com",
    "nature.com",
    "who.int",
]

# Tool definitions for LiteLLM function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_reputable_sources",
            "description": (
                "Search reputable news outlets, fact-checking sites, and "
                "academic/government sources (Reuters, AP News, BBC, FactCheck.org, "
                "PolitiFact, Snopes, WHO, etc.) for evidence about a claim. "
                "Always use this tool first."
            ),
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
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Unrestricted general web search. Use as a fallback only when "
                "reputable sources do not provide sufficient information to reach a verdict."
            ),
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


def search_reputable_sources(query: str, max_results: int = 6) -> str:
    """Search within reputable sites using a site-filter query."""
    site_filter = " OR ".join(f"site:{s}" for s in REPUTABLE_SITES)
    restricted_query = f"({query}) ({site_filter})"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(restricted_query, max_results=max_results))
        if not results:
            # Retry without strict site filter (DDG may reject complex queries)
            with DDGS() as ddgs:
                results = list(ddgs.text(query + " site:reuters.com OR site:apnews.com OR site:bbc.com OR site:factcheck.org OR site:politifact.com", max_results=max_results))
        return _format_results(results)
    except Exception as e:
        return f"Search error: {e}"


def search_web(query: str, max_results: int = 6) -> str:
    """Unrestricted DuckDuckGo web search."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return _format_results(results)
    except Exception as e:
        return f"Search error: {e}"


def _dispatch_tool(name: str, arguments: dict) -> str:
    """Route a tool call to the appropriate function."""
    if name == "search_reputable_sources":
        print(f"  [tool] search_reputable_sources: {arguments.get('query', '')!r}")
        return search_reputable_sources(arguments["query"])
    if name == "search_web":
        print(f"  [tool] search_web: {arguments.get('query', '')!r}")
        return search_web(arguments["query"])
    return f"Unknown tool: {name}"


SYSTEM_PROMPT = """\
You are a rigorous, non-partisan fact-checker specializing in political claims.

Your task:
1. Use `search_reputable_sources` first to find evidence from authoritative outlets.
2. If those results are insufficient, use `search_web` as a fallback.
3. Cross-reference multiple sources before reaching a verdict.
4. Reason carefully about the evidence — distinguish between confirmed facts,
   contested claims, and outright falsehoods.

After gathering sufficient evidence, respond ONLY with a JSON object (no markdown fences):
{
  "verdict": "<TRUE|MOSTLY_TRUE|MISLEADING|MOSTLY_FALSE|FALSE|UNVERIFIABLE>",
  "confidence": <0.0–1.0>,
  "summary": "<2–3 sentence summary of findings>",
  "justification": "<detailed explanation with specific evidence>",
  "sources": ["<URL or source name>", ...]
}

Verdict definitions:
- TRUE: Claim is accurate and well-supported.
- MOSTLY_TRUE: Claim is largely accurate but omits context or has minor inaccuracies.
- MISLEADING: Claim uses true facts in a deceptive or out-of-context way.
- MOSTLY_FALSE: Claim has a grain of truth but is substantially inaccurate.
- FALSE: Claim is factually incorrect.
- UNVERIFIABLE: Insufficient credible evidence to determine truth or falsehood.
"""


def verify_claim(claim: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """
    Verify a political claim using an LLM with web-search tool calling.

    Performs an agentic loop: the LLM calls search tools as needed,
    then produces a structured JSON verdict.

    Args:
        claim: The political fact claim to verify.
        model: LiteLLM model string (e.g. "openai/gpt-4o-mini").

    Returns:
        Dict with keys: verdict, confidence, summary, justification, sources.
    """
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Please fact-check this claim:\n\n\"{claim}\""},
    ]

    for iteration in range(12):  # safety cap on agentic loop iterations
        response = litellm.completion(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_msg = response.choices[0].message

        # Append assistant message — use dict form for clean serialisation
        msg_dict: dict[str, Any] = {"role": "assistant"}
        if assistant_msg.content:
            msg_dict["content"] = assistant_msg.content
        if assistant_msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_msg.tool_calls
            ]
        messages.append(msg_dict)

        if assistant_msg.tool_calls:
            # Execute each tool call and feed results back
            for tc in assistant_msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result = _dispatch_tool(tc.function.name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
        else:
            # No tool calls → LLM has produced its final answer
            content = (assistant_msg.content or "").strip()
            try:
                start = content.find("{")
                end = content.rfind("}") + 1
                if start != -1 and end > start:
                    return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
            # Fallback: wrap raw content
            return {
                "verdict": "UNVERIFIABLE",
                "confidence": 0.0,
                "summary": content,
                "justification": content,
                "sources": [],
            }

    return {
        "verdict": "UNVERIFIABLE",
        "confidence": 0.0,
        "summary": "Reached maximum iterations without a conclusive verdict.",
        "justification": "The agent could not complete the fact-check within the allowed steps.",
        "sources": [],
    }



