"""
Streamlit web interface for MetaFact — political claim fact-checker.

Run with:
    streamlit run app.py
"""

import io
import os
import sys

import litellm
import streamlit as st

from factcheck import verify_claim

VERDICT_CONFIG = {
    "TRUE":         {"color": "#2e7d32", "icon": "✅", "label": "True"},
    "MOSTLY_TRUE":  {"color": "#558b2f", "icon": "✔️",  "label": "Mostly True"},
    "MISLEADING":   {"color": "#e65100", "icon": "⚠️",  "label": "Misleading"},
    "MOSTLY_FALSE": {"color": "#c62828", "icon": "❌",  "label": "Mostly False"},
    "FALSE":        {"color": "#b71c1c", "icon": "🚫",  "label": "False"},
    "UNVERIFIABLE": {"color": "#546e7a", "icon": "❓",  "label": "Unverifiable"},
}

DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-4.1",
    "anthropic/claude-3-5-haiku-latest",
    "anthropic/claude-3-7-sonnet-latest",
]


def _render_verdict(result: dict) -> None:
    verdict = result.get("verdict", "UNVERIFIABLE")
    cfg = VERDICT_CONFIG.get(verdict, VERDICT_CONFIG["UNVERIFIABLE"])
    confidence = result.get("confidence", 0.0)

    st.markdown(
        f"""
        <div style="
            background:{cfg['color']}22;
            border-left: 6px solid {cfg['color']};
            border-radius: 6px;
            padding: 1rem 1.4rem;
            margin-bottom: 1rem;
        ">
            <span style="font-size:2rem;">{cfg['icon']}</span>
            <span style="font-size:1.5rem; font-weight:700; color:{cfg['color']}; margin-left:0.5rem;">
                {cfg['label']}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"**Confidence:** {confidence:.0%}")
    st.progress(confidence)


def main() -> None:
    st.set_page_config(
        page_title="MetaFact — Fact Checker",
        page_icon="🔍",
        layout="centered",
    )

    st.title("🔍 MetaFact")
    st.caption("AI-powered political fact-checker using web search and LLMs.")

    # ── Sidebar: settings ────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        model = st.selectbox(
            "Model",
            options=DEFAULT_MODELS,
            index=0,
            help="LiteLLM model string. You can also type a custom model.",
        )
        custom_model = st.text_input(
            "Custom model (overrides above)",
            placeholder="e.g. groq/llama3-70b-8192",
        )
        if custom_model.strip():
            model = custom_model.strip()

        api_key = st.text_input(
            "API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Set OPENAI_API_KEY env var to pre-fill.",
        )
        api_base = st.text_input(
            "API Base URL (optional)",
            value=os.getenv("OPENAI_API_BASE", ""),
            placeholder="https://api.openai.com/v1",
        )

        st.divider()
        st.markdown(
            "Sources searched: Reuters, AP News, BBC, FactCheck.org, "
            "PolitiFact, Snopes, WHO, and more."
        )

    # ── Main area ─────────────────────────────────────────────────────────────
    claim = st.text_area(
        "Enter a claim to fact-check",
        placeholder="e.g. The US has the highest GDP per capita in the world.",
        height=120,
    )

    run = st.button("🔍 Fact-check", type="primary", disabled=not claim.strip())

    if run and claim.strip():
        # Apply API settings
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if api_base:
            litellm.api_base = api_base

        tool_log: list[str] = []

        # Capture print() output from factcheck.py to show tool activity
        old_stdout = sys.stdout
        buf = io.StringIO()

        with st.status("Researching claim…", expanded=True) as status:
            log_placeholder = st.empty()

            class _LogCapture(io.StringIO):
                def write(self, text: str) -> int:  # type: ignore[override]
                    result = buf.write(text)
                    old_stdout.write(text)
                    stripped = text.strip()
                    if stripped:
                        tool_log.append(stripped)
                        log_placeholder.markdown(
                            "\n".join(f"- {line}" for line in tool_log[-10:])
                        )
                    return result

            sys.stdout = _LogCapture()
            try:
                result = verify_claim(claim.strip(), model=model)
                status.update(label="Done ✓", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="Error", state="error", expanded=True)
                st.error(f"**Error:** {exc}")
                result = None
            finally:
                sys.stdout = old_stdout

        if result:
            st.divider()
            _render_verdict(result)

            st.subheader("Summary")
            st.write(result.get("summary", "—"))

            with st.expander("📄 Full Justification"):
                st.write(result.get("justification", "—"))

            sources = result.get("sources", [])
            if sources:
                with st.expander(f"🔗 Sources ({len(sources)})"):
                    for src in sources:
                        if src.startswith("http"):
                            st.markdown(f"- [{src}]({src})")
                        else:
                            st.markdown(f"- {src}")


if __name__ == "__main__":
    main()
