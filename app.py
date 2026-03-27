"""
app.py
──────
Simple Streamlit UI to test the RAG chatbot pipeline.
Run from project root: streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Support RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean background */
    .stApp { background-color: #0f1117; }

    /* Input box */
    .stTextInput > div > div > input {
        background-color: #1e2130;
        color: #ffffff;
        border: 1px solid #3d4466;
        border-radius: 8px;
        padding: 12px;
        font-size: 15px;
    }

    /* Answer box */
    .answer-box {
        background-color: #1a1d2e;
        border-left: 4px solid #4f8ef7;
        border-radius: 8px;
        padding: 20px 24px;
        margin-top: 16px;
        color: #e8eaf0;
        font-size: 15px;
        line-height: 1.7;
    }

    /* Chunk card */
    .chunk-card {
        background-color: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 13px;
        color: #b0b8d0;
        line-height: 1.6;
    }

    .chunk-meta {
        color: #4f8ef7;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .score-badge {
        display: inline-block;
        background-color: #2a3560;
        color: #7eb3ff;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 11px;
        margin-left: 8px;
    }

    /* Header */
    h1 { color: #ffffff !important; }
    h3 { color: #8892b0 !important; font-size: 13px !important;
         text-transform: uppercase; letter-spacing: 1px; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load pipeline (cached so it loads once) ───────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG pipeline...")
def load_pipeline():
    from src.vector_store.search import VectorSearcher
    from src.rag.generator import Generator
    searcher  = VectorSearcher("config/config.yaml")
    generator = Generator()
    return searcher, generator

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🤖 Support RAG Chatbot")
st.markdown("Ask any customer support question and get an answer from the knowledge base.")
st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
query = st.text_input(
    label="Your question",
    placeholder="e.g. How do I cancel my order?",
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    ask_btn = st.button("Ask ✦", use_container_width=True, type="primary")
with col2:
    show_chunks = st.checkbox("Show chunks", value=False)

# ── Run pipeline ──────────────────────────────────────────────────────────────
if ask_btn and query.strip():
    try:
        searcher, generator = load_pipeline()

        with st.spinner("Retrieving relevant knowledge..."):
            results, context = searcher.search_and_format(query.strip(), top_k=5)

        with st.spinner("Generating answer..."):
            answer = generator.generate(query.strip(), context)

        # ── Answer ────────────────────────────────────────────────────────
        st.markdown("### Answer")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        # ── Retrieved chunks (optional) ────────────────────────────────────
        if show_chunks and results:
            st.markdown("### Retrieved Chunks")
            for i, chunk in enumerate(results, 1):
                score   = chunk.get("score", 0)
                intent  = chunk.get("intent", "")
                source  = chunk.get("source", "")
                instruction = chunk.get("instruction", "")[:120]
                response    = chunk.get("response", "")[:200]

                st.markdown(f"""
                <div class="chunk-card">
                    <div class="chunk-meta">
                        #{i} · {source} · {intent}
                        <span class="score-badge">score {score:.4f}</span>
                    </div>
                    <b>Q:</b> {instruction}...<br>
                    <b>A:</b> {response}...
                </div>
                """, unsafe_allow_html=True)

    except RuntimeError as e:
        if "Ollama" in str(e):
            st.error("⚠️ Ollama is not running. Start it with: `ollama serve`")
        else:
            st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

elif ask_btn and not query.strip():
    st.warning("Please enter a question first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("RAG · BGE-base-en-v1.5 · Mistral 7B · MongoDB")