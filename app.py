"""
app.py
──────
Streamlit UI for the Customer Support RAG Chatbot.

Run:
    streamlit run app.py

Calls the Modal Cloud deployment (deploy/modal/app.py) for retrieval and
generation — no local services required.
"""

import html

import requests
import streamlit as st

MODAL_APP_NAME = "support-rag-chatbot"
MODAL_WORKSPACE = "alhasanmuhammadai"
CATEGORIES = [
    "All", "ACCOUNT", "CANCEL", "CONTACT", "DELIVERY", "FEEDBACK",
    "INVOICE", "ORDER", "PAYMENT", "REFUND", "SHIPPING", "SUBSCRIPTION",
]
SAMPLE_QUESTIONS = [
    "How do I cancel my order?",
    "What's your refund policy?",
    "How can I update my shipping address?",
    "What payment methods do you accept?",
    "How do I create an account?",
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Support RAG Chatbot",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Theme / styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --surface: #FFFFFF;
        --surface-alt: #F1ECE3;
        --border: #E3DDD2;
        --text: #2B2A28;
        --text-muted: #8A8580;
        --accent: #2F5D50;
        --accent-soft: #E6EEEA;
        --accent-text: #1F4338;
        --gold: #8A6D3B;
        --gold-soft: #F4ECDD;
        --danger: #B5483B;
    }

    /* Hero */
    .hero-eyebrow {
        font-size: 11px; font-weight: 700; letter-spacing: 0.16em;
        text-transform: uppercase; color: var(--accent); margin-bottom: 6px;
    }
    .hero-title {
        font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em;
        color: var(--text); line-height: 1.2; margin-bottom: 6px;
    }
    .hero-subtitle { color: var(--text-muted); font-size: 0.95rem; margin-bottom: 0.5rem; }

    /* Status pills */
    .status-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 4px 0; }
    .status-pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
        border: 1px solid var(--border); background: var(--surface); color: var(--text-muted);
    }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    .status-dot.on  { background: var(--accent); }
    .status-dot.off { background: var(--danger); }

    /* Source cards */
    .source-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 12px; padding: 12px 16px; margin-bottom: 10px;
    }
    .source-card-header {
        display: flex; justify-content: space-between; align-items: center;
        gap: 8px; margin-bottom: 8px; flex-wrap: wrap;
    }
    .source-tag {
        font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
        color: var(--accent-text); background: var(--accent-soft);
        padding: 2px 8px; border-radius: 6px;
    }
    .source-score {
        font-size: 11px; font-weight: 700; color: var(--gold);
        background: var(--gold-soft); padding: 2px 8px; border-radius: 6px;
    }
    .source-q { font-size: 13.5px; color: var(--text); margin-bottom: 4px; line-height: 1.5; }
    .source-a { font-size: 13.5px; color: var(--text-muted); line-height: 1.5; }

    /* Tech footer */
    .tech-row { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin-top: 4px; }
    .tech-badge {
        font-size: 11.5px; font-weight: 600; color: var(--text-muted);
        background: var(--surface-alt); border: 1px solid var(--border);
        padding: 4px 10px; border-radius: 999px;
    }
    .footer-caption { text-align: center; color: var(--text-muted); font-size: 12px; margin-top: 10px; }

    /* Sidebar polish */
    section[data-testid="stSidebar"] .stButton button { border-radius: 8px; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Status checks ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=8, show_spinner=False)
def check_modal_status(workspace: str):
    if not workspace:
        return False
    try:
        r = requests.get(_modal_url(workspace, "health"), timeout=5)
        return r.ok
    except requests.RequestException:
        return False


def _modal_url(workspace: str, fn_name: str) -> str:
    return f"https://{workspace}--{MODAL_APP_NAME}-{fn_name}.modal.run"


def _modal_api_key() -> str:
    try:
        return st.secrets.get("MODAL_API_KEY", "")
    except Exception:
        return ""


def status_pill(label: str, online: bool) -> str:
    state = "on" if online else "off"
    text = "online" if online else "offline"
    return (
        f'<span class="status-pill"><span class="status-dot {state}"></span>'
        f"{label} · {text}</span>"
    )


# ── Backend ──────────────────────────────────────────────────────────────────

def run_modal(query: str, top_k: int, category: str, workspace: str, api_key: str) -> dict:
    payload = {"query": query, "top_k": top_k}
    if category != "All":
        payload["category"] = category

    headers = {"X-API-Key": api_key} if api_key else {}
    resp = requests.post(_modal_url(workspace, "chat"), json=payload, headers=headers, timeout=180)
    if resp.status_code == 401:
        raise RuntimeError("Authentication with the assistant service failed.")
    resp.raise_for_status()
    return resp.json()


def friendly_error(e: Exception) -> str:
    return f"Couldn't reach the assistant right now: {e}"


def render_chunk_card(chunk: dict, idx: int) -> str:
    score = chunk.get("score", 0)
    intent = html.escape(chunk.get("intent", "general").replace("_", " ").title())
    category = html.escape(chunk.get("category", ""))
    instruction = html.escape(chunk.get("instruction", ""))
    response = html.escape(chunk.get("response", ""))

    if len(instruction) > 160:
        instruction = instruction[:160].rstrip() + "…"
    if len(response) > 240:
        response = response[:240].rstrip() + "…"

    return f"""
    <div class="source-card">
      <div class="source-card-header">
        <span class="source-tag">#{idx} · {category} · {intent}</span>
        <span class="source-score">match {score:.2f}</span>
      </div>
      <div class="source-q"><strong>Q —</strong> {instruction}</div>
      <div class="source-a"><strong>A —</strong> {response}</div>
    </div>
    """


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ✨ Support RAG")
    st.caption("Retrieval-augmented customer support assistant")
    st.divider()

    workspace = MODAL_WORKSPACE
    api_key = _modal_api_key()
    online = check_modal_status(workspace)
    st.markdown(
        '<div class="status-row">' + status_pill("Assistant", online) + "</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("**Retrieval settings**")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)
    category = st.selectbox("Category filter", CATEGORIES)

    st.divider()
    st.markdown("**Try asking**")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero-eyebrow">Customer Support · RAG</div>
    <div class="hero-title">Ask anything about your order</div>
    <div class="hero-subtitle">
        Answers are grounded in our support knowledge base — semantic retrieval,
        cross-encoder reranking, and Mistral 7B.
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ── Chat history ────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        sources = msg.get("sources")
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for i, chunk in enumerate(sources, 1):
                    st.markdown(render_chunk_card(chunk, i), unsafe_allow_html=True)

# ── Input ────────────────────────────────────────────────────────────────────────

query = st.chat_input("Ask a customer support question…")
if not query and "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    st.session_state.messages.append({"role": "user", "content": query, "avatar": "🧑"})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Thinking…"):
            try:
                result = run_modal(query, top_k, category, workspace, api_key)

                answer = result.get("answer", "")
                sources = result.get("retrieved_chunks", [])

                st.markdown(answer)
                if sources:
                    with st.expander(f"Sources ({len(sources)})"):
                        for i, chunk in enumerate(sources, 1):
                            st.markdown(render_chunk_card(chunk, i), unsafe_allow_html=True)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "avatar": "✨", "sources": sources}
                )
            except Exception as e:
                err = friendly_error(e)
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ {err}", "avatar": "✨"}
                )

# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="tech-row">
        <span class="tech-badge">BGE-base-en-v1.5</span>
        <span class="tech-badge">MongoDB Vector Search</span>
        <span class="tech-badge">Cross-Encoder Reranker</span>
        <span class="tech-badge">Mistral 7B · Ollama</span>
        <span class="tech-badge">Modal Serverless</span>
    </div>
    <div class="footer-caption">Customer Support RAG Chatbot · DEPI Graduation Project</div>
    """,
    unsafe_allow_html=True,
)
