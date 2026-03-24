#!/usr/bin/env python3
"""
Fact-checker CLI for political claims using LiteLLM and web search.

Usage:
    python cli.py '<claim>'
    LITELLM_MODEL=openai/gpt-4o python cli.py '<claim>'

Environment variables:
    LITELLM_MODEL       LiteLLM model string (default: openai/gpt-4o-mini)
    OPENAI_API_KEY      OpenAI API key (or any OpenAI-compatible key)
    OPENAI_API_BASE     Optional custom API base URL for OpenAI-compatible endpoints
"""

import os
import sys
from typing import Any

import litellm

from factcheck import verify_claim


def main(claim: str) -> dict[str, Any]:
    """Entry point: fact-check *claim* and print a formatted report."""
    model = os.getenv("LITELLM_MODEL", "openai/gpt-4o-mini")

    # Allow overriding the API base (for OpenAI-compatible endpoints)
    api_base = os.getenv("OPENAI_API_BASE")
    if api_base:
        litellm.api_base = api_base

    print(f"Claim    : {claim!r}")
    print(f"Model    : {model}")
    print()

    result = verify_claim(claim, model=model)

    print()
    print("=" * 60)
    verdict = result.get("verdict", "N/A")
    confidence = result.get("confidence", 0.0)
    bar = "█" * round(confidence * 20)
    print(f"Verdict    : {verdict}")
    print(f"Confidence : {confidence:.0%}  [{bar:<20}]")
    print()
    print(f"Summary:\n  {result.get('summary', 'N/A')}")
    print()
    print(f"Justification:\n  {result.get('justification', 'N/A')}")
    print()
    sources = result.get("sources", [])
    if sources:
        print("Sources:")
        for src in sources:
            print(f"  • {src}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(" ".join(sys.argv[1:]))
