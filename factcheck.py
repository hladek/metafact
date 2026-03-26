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

# Fallback sources used only if the LLM fails to produce relevant ones
_FALLBACK_SITES = [
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
                "Search the pre-selected reputable sources identified for this claim "
                "(topic-relevant news outlets, fact-checking sites, academic/government "
                "sources). Always use this tool first."
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


def get_relevant_sources(claim: str, model: str) -> list[str]:
    """Ask the LLM for 10 reputable domains most relevant to fact-checking this claim."""
    response = litellm.completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research librarian. Given a claim, return the 10 most "
                    "reputable and relevant domains (news outlets, fact-checking sites, "
                    "academic or government sources) for verifying it. "
                    "Respond ONLY with a JSON array of domain names, e.g.: "
                    '["reuters.com", "who.int"]'
                ),
            },
            {
                "role": "user",
                "content": (
                    f'Claim: "{claim}"\n\n'
                    "List 10 reputable domains best suited to fact-check this claim."
                ),
            },
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    try:
        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end > start:
            domains = json.loads(content[start:end])
            sites = [d.strip() for d in domains if isinstance(d, str)][:10]
            if sites:
                return sites
    except (json.JSONDecodeError, ValueError):
        pass
    return _FALLBACK_SITES


def search_reputable_sources(query: str, sites: list[str], max_results: int = 6) -> str:
    """Search within the provided reputable sites using a site-filter query."""
    site_filter = " OR ".join(f"site:{s}" for s in sites)
    restricted_query = f"({query}) ({site_filter})"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(restricted_query, max_results=max_results))
        if not results:
            # Retry with just the first three sites (DDG may reject long queries)
            short_filter = " OR ".join(f"site:{s}" for s in sites[:3])
            with DDGS() as ddgs:
                results = list(ddgs.text(f"({query}) ({short_filter})", max_results=max_results))
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


def _dispatch_tool(name: str, arguments: dict, sites: list[str]) -> str:
    """Route a tool call to the appropriate function."""
    if name == "search_reputable_sources":
        print(f"  [tool] search_reputable_sources: {arguments.get('query', '')!r}")
        return search_reputable_sources(arguments["query"], sites=sites)
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
    # Step 1: identify the best sources for this specific claim
    print("  [setup] Identifying relevant sources…")
    sites = get_relevant_sources(claim, model)
    print(f"  [setup] Sources: {', '.join(sites)}")

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
                result = _dispatch_tool(tc.function.name, args, sites)
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



