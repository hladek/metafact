"""
Streamlit web interface for MetaFact — political claim fact-checker.

Run with:
    streamlit run app.py
"""

import io
import sys

import streamlit as st

from factcheck import verify_claim_model, verify_claim_search

VERDICT_CONFIG = {
    "TRUE":         {"color": "#2e7d32", "icon": "✅", "label": "True"},
    "MOSTLY_TRUE":  {"color": "#558b2f", "icon": "✔️",  "label": "Mostly True"},
    "MISLEADING":   {"color": "#e65100", "icon": "⚠️",  "label": "Misleading"},
    "MOSTLY_FALSE": {"color": "#c62828", "icon": "❌",  "label": "Mostly False"},
    "FALSE":        {"color": "#b71c1c", "icon": "🚫",  "label": "False"},
    "UNVERIFIABLE": {"color": "#546e7a", "icon": "❓",  "label": "Unverifiable"},
}

_CSS = """
<style>
    /* Page background & typography */
    [data-testid="stAppViewContainer"] {
        background: #f8f9fb;
    }
    [data-testid="stMain"] > div {
        padding-top: 2.5rem;
    }
    h1 { letter-spacing: -0.5px; }

    /* Subtle text area */
    textarea {
        border-radius: 10px !important;
        font-size: 1rem !important;
    }

    /* Round primary button */
    [data-testid="stBaseButton-primary"] button,
    button[kind="primary"] {
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* Expander polish */
    [data-testid="stExpander"] {
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
    }

    /* Hide the Streamlit footer */
    footer { visibility: hidden; }
</style>
"""


def _render_verdict(result: dict) -> None:
    verdict = result.get("verdict", "UNVERIFIABLE")
    cfg = VERDICT_CONFIG.get(verdict, VERDICT_CONFIG["UNVERIFIABLE"])
    confidence = result.get("confidence", 0.0)
    bar_color = cfg["color"]
    pct = int(confidence * 100)

    st.markdown(
        f"""
        <div style="
            background: {cfg['color']}18;
            border: 1px solid {cfg['color']}55;
            border-radius: 10px;
            padding: 1.1rem 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.25rem;
        ">
            <span style="font-size:1.6rem; line-height:1;">
                {cfg['icon']}
                <span style="
                    font-size:1.35rem;
                    font-weight:700;
                    color:{cfg['color']};
                    margin-left:0.5rem;
                    vertical-align:middle;
                ">{cfg['label']}</span>
            </span>
            <span style="
                font-size:1.1rem;
                font-weight:600;
                color:{cfg['color']};
                opacity:0.85;
            ">{pct}% confidence</span>
        </div>
        <div style="
            height: 5px;
            border-radius: 0 0 6px 6px;
            background: #e0e0e0;
            margin-bottom: 1.2rem;
            overflow: hidden;
        ">
            <div style="
                width: {pct}%;
                height: 100%;
                background: {bar_color};
                border-radius: 0 0 6px 6px;
                transition: width 0.4s ease;
            "></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="MetaFact — Fact Checker",
        page_icon="🔍",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.markdown(_CSS, unsafe_allow_html=True)

    st.title("🔍 MetaFact")
    st.caption("AI-powered political claim fact-checker · LLM · Web search")
    st.divider()

    claim = st.text_area(
        "Enter a claim to fact-check",
        placeholder="e.g. The US has the highest GDP per capita in the world.",
        height=110,
        label_visibility="collapsed",
    )

    col1, col2, _ = st.columns([2, 2.5, 3])
    with col1:
        run = st.button("🔍 Fact-check", type="primary", use_container_width=True, disabled=not claim.strip())
    with col2:
        run_search = st.button("🌐 Search & verify", use_container_width=True, disabled=not claim.strip())

    checker_fn = None
    status_label = ""
    if run and claim.strip():
        checker_fn = verify_claim_model
        status_label = "Analysing claim…"
    elif run_search and claim.strip():
        checker_fn = verify_claim_search
        status_label = "Searching the web…"

    if checker_fn:
        tool_log: list[str] = []

        # Capture print() output from factcheck.py to show tool activity
        old_stdout = sys.stdout
        buf = io.StringIO()

        with st.status(status_label, expanded=True) as status:
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
                result = checker_fn(claim.strip())
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

            st.markdown("**Summary**")
            st.write(result.get("summary", "—"))

            with st.expander("📄 Full justification"):
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
