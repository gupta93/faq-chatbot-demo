from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag.answer import AnswerResult, answer_question
from rag.llm import load_llm_config
from rag.retriever import DocumentRetriever

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"


@st.cache_resource
def get_retriever() -> DocumentRetriever:
    return DocumentRetriever(data_dir=DATA_DIR, min_score=0.08)


def render_citations(result: AnswerResult) -> None:
    if not result.citations:
        return
    st.markdown("**Sources used**")
    for c in result.citations:
        st.caption(f"📄 `{c.doc_id}` — *{c.section}* (relevance {c.score:.2f})")


def main() -> None:
    st.set_page_config(
        page_title="FAQ Assistant Demo",
        page_icon="☕",
        layout="centered",
    )
    st.title("☕ Document-Grounded FAQ Assistant")
    st.markdown(
        """
        This demo answers customer questions **only from your business documents**.
        If nothing relevant is found, it says *I don't know* instead of guessing.

        Sample content is a **fictional** coffee subscription brand in the `data/` folder.
        """
    )

    config = load_llm_config()
    if config.mock:
        st.info("Running in **mock mode** (`MOCK_LLM=1`) — no API key required.")
    elif not config.api_key:
        st.warning(
            "Set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`, `MODEL`) in `.env` "
            "to use a live model. OpenRouter is supported by default."
        )

    with st.expander("How it works"):
        st.markdown(
            """
            1. Your documents live in **`data/`** (Markdown, plain text, or PDF).
            2. Questions are matched to the most relevant passages (TF-IDF retrieval).
            3. An AI model answers using **only** those passages, with source citations.
            4. Low-confidence matches are refused with *I don't know*.
            """
        )

    retriever = get_retriever()
    st.caption(f"Indexed {retriever.chunk_count} passages from `{DATA_DIR.name}/`.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                for line in msg["citations"]:
                    st.caption(line)

    if prompt := st.chat_input("Ask a question about subscriptions, shipping, billing…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents…"):
                result = answer_question(prompt, retriever, config=config)
            st.markdown(result.text)
            render_citations(result)
            citation_lines = [
                f"`{c.doc_id}` — {c.section} ({c.score:.2f})" for c in result.citations
            ]
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.text,
                    "citations": citation_lines,
                }
            )

    st.sidebar.header("Try these questions")
    examples = [
        "How much is the Enthusiast plan?",
        "Can I pause my subscription?",
        "Do you ship to Canada?",
        "What is the capital of France?",
    ]
    for ex in examples:
        if st.sidebar.button(ex, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": ex})
            result = answer_question(ex, retriever, config=config)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.text,
                    "citations": [
                        f"`{c.doc_id}` — {c.section} ({c.score:.2f})"
                        for c in result.citations
                    ],
                }
            )
            st.rerun()


if __name__ == "__main__":
    main()
