"""
app.py — Streamlit UI for the free PDF RAG system
100% free stack: sentence-transformers + ChromaDB + HuggingFace / Ollama / Extractive fallback
"""

import streamlit as st
import os
from rag import RAGPipeline

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A — Free RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0d1117; }
  [data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }

  .chat-user {
    background: #1c2128;
    border-left: 3px solid #58a6ff;
    padding: 12px 16px; border-radius: 8px; margin: 8px 0;
    color: #c9d1d9;
  }
  .chat-bot {
    background: #0d2137;
    border-left: 3px solid #3fb950;
    padding: 12px 16px; border-radius: 8px; margin: 8px 0;
    color: #c9d1d9;
  }
  .source-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px; padding: 10px 14px;
    font-size: 0.81em; color: #8b949e; margin: 4px 0;
  }
  .backend-badge {
    display: inline-block;
    background: #1f2d1f; color: #3fb950;
    border: 1px solid #238636;
    border-radius: 20px; padding: 2px 10px;
    font-size: 0.78em; margin-left: 8px;
  }
  .info-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 14px 18px; margin: 8px 0;
  }
  .stButton>button {
    background: #238636; color: #fff;
    border: none; border-radius: 8px;
    font-weight: 600; width: 100%;
  }
  .stButton>button:hover { background: #2ea043; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [
    ("rag", None), ("chat", []), ("loaded", False), ("stats", {}),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 LLM Backend")

    st.markdown("""
<div class="info-card">
<b>Three free options — tried in order:</b><br><br>
🟢 <b>Ollama</b> — best quality, fully local<br>
🟡 <b>HuggingFace API</b> — free tier, needs token<br>
⚪ <b>Extractive</b> — always works, no token
</div>
""", unsafe_allow_html=True)

    hf_token = st.text_input(
        "HuggingFace Token (optional)",
        type="password",
        placeholder="hf_...",
        help="Free at huggingface.co/settings/tokens — enables flan-t5-large answers",
    )
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    st.markdown("---")
    st.markdown("## ⚙️ Chunking")
    chunk_size    = st.slider("Chunk size",    100, 800, 400, 50)
    chunk_overlap = st.slider("Chunk overlap",   0, 150,  60, 10)
    top_k         = st.slider("Top-K chunks",    1,   8,   4)

    st.markdown("---")
    st.markdown("## 📊 Document Info")
    if st.session_state.stats:
        s = st.session_state.stats
        c1, c2 = st.columns(2)
        c1.metric("Pages",  s.get("pages",  "—"))
        c2.metric("Chunks", s.get("chunks", "—"))
        st.caption(f"~{s.get('words','—'):,} words")
    else:
        st.caption("No document loaded yet")

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat = []
        st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 📄 PDF Question Answering")
st.markdown("**100% free · No paid APIs · Runs on CPU**")

st.markdown("""
<div class="info-card">
<b>Free Stack:</b>
&nbsp; 📦 <code>PyMuPDF</code> — PDF extraction &nbsp;|&nbsp;
🧠 <code>all-MiniLM-L6-v2</code> — free local embeddings &nbsp;|&nbsp;
🗄️ <code>ChromaDB</code> — vector database &nbsp;|&nbsp;
🤖 <code>Ollama / HF API / Extractive</code> — answer generation
</div>
""", unsafe_allow_html=True)

st.markdown("---")
left, right = st.columns([1, 1], gap="large")

# ── LEFT: Upload ───────────────────────────────────────────────────────────────
with left:
    st.markdown("### 📤 Upload PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded:
        st.success(f"**{uploaded.name}** — {round(uploaded.size/1024, 1)} KB")

        if st.button("🚀 Process & Index"):
            with st.spinner("Extracting → chunking → embedding (first run downloads ~90 MB model)…"):
                try:
                    rag = RAGPipeline(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        hf_token=hf_token,
                    )
                    stats = rag.ingest(uploaded)
                    st.session_state.rag    = rag
                    st.session_state.loaded = True
                    st.session_state.stats  = stats
                    st.session_state.chat   = []
                    st.success(f"✅ Indexed **{stats['chunks']} chunks** from **{stats['pages']} pages**!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")

    st.markdown("### 💡 Try These Questions")
    examples = [
        "What is this document about?",
        "Summarize the main points.",
        "What are the key conclusions?",
        "List any important numbers or dates.",
        "Who wrote this document?",
        "What problem does this document address?",
    ]
    for q in examples:
        if st.button(q, key=f"ex_{q}"):
            if st.session_state.loaded:
                st.session_state._auto_q = q
                st.rerun()
            else:
                st.warning("Process a PDF first.")

    # Ollama setup tip
    with st.expander("🦙 Want better answers? Install Ollama (free)"):
        st.markdown("""
**Ollama runs LLMs 100% locally — no internet needed after install.**

```bash
# Linux / Codespaces
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral        # ~4 GB download
ollama serve               # keep running in another terminal
```

Then re-index your PDF and ask questions — answers will be much better!
        """)

    with st.expander("🤗 HuggingFace free token setup"):
        st.markdown("""
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → Role: **Read** → Generate
3. Paste it in the sidebar field above

This enables **flan-t5-large** for better answers — completely free.
        """)

# ── RIGHT: Chat ────────────────────────────────────────────────────────────────
with right:
    st.markdown("### 💬 Ask Your Document")

    if not st.session_state.loaded:
        st.info("👈 Upload and process a PDF to start asking questions.")
    else:
        # Chat history
        for msg in st.session_state.chat:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">🧑&nbsp; {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                backend_badge = f'<span class="backend-badge">via {msg.get("backend","?")}</span>'
                st.markdown(
                    f'<div class="chat-bot">🤖&nbsp; {msg["content"]}{backend_badge}</div>',
                    unsafe_allow_html=True,
                )
                if msg.get("sources"):
                    with st.expander(f"📎 {len(msg['sources'])} source chunks"):
                        for i, src in enumerate(msg["sources"], 1):
                            st.markdown(
                                f'<div class="source-box"><b>#{i}</b> {src}</div>',
                                unsafe_allow_html=True,
                            )

        default_q = st.session_state.pop("_auto_q", "")
        question  = st.text_input(
            "Your question",
            value=default_q,
            placeholder="Ask anything about the PDF…",
        )

        if st.button("🔍 Get Answer") and question.strip():
            with st.spinner("Searching document + generating answer…"):
                try:
                    result = st.session_state.rag.query(question, top_k=top_k)
                    st.session_state.chat.append({"role": "user", "content": question})
                    st.session_state.chat.append({
                        "role":    "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "backend": result["backend"],
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>100% free · PyMuPDF · sentence-transformers · ChromaDB · "
    "Ollama / HuggingFace / Extractive</small></center>",
    unsafe_allow_html=True,
)
