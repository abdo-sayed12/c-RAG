import os
import pickle
import faiss
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from groq import Groq
import streamlit as st

# ─── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")
LLM_MODEL_NAME    = os.environ.get("LLM_MODEL_NAME", "llama-3.3-70b-versatile")
EMBEDDING_MODEL   = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL    = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")
FAISS_INDEX_PATH  = os.environ.get("FAISS_INDEX_PATH", "data/processed/faiss.index")
CHUNKS_PKL_PATH   = os.environ.get("CHUNKS_PKL_PATH", "data/processed/chunks.pkl")

CONDITION_MAP = {
    "stroke": "3.1",
    "spinal cord": "3.2",
    "amputation": "3.3",
    "fracture": "3.4",
    "low back pain": "3.5",
    "copd": "3.6",
    "cardiac": "3.7",
    "burns": "3.8",
}

SYSTEM_PROMPT = (
    "You are Care360, an advanced AI clinical assistant. "
    "Answer the user's question using only the provided WHO 'Package of interventions for rehabilitation' evidence. "
    "Do not hallucinate or include outside information. "
    "Frame all recommendations as AI suggestions derived from WHO guidelines, e.g. "
    "'Based on the WHO guidelines, I suggest...'. "
    "Always cite sources using: (Source: [Section Name], Section [Section Number])."
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Care360 — Clinical AI",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f1117; }
    .source-card {
        background: #1e2130;
        border-left: 3px solid #4f8ef7;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13px;
    }
    .source-title { color: #4f8ef7; font-weight: 600; margin-bottom: 4px; }
    .source-text  { color: #9ba3af; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ─── Model loading (cached — runs once) ───────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    tok_emb    = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
    mod_emb    = AutoModel.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
    tok_rerank = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    mod_rerank = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
    mod_rerank.eval()
    return tok_emb, mod_emb, tok_rerank, mod_rerank

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_index():
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PKL_PATH):
        idx = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_PKL_PATH, "rb") as f:
            cks = pickle.load(f)
        return idx, cks
    return None, []

tok_emb, mod_emb, tok_rerank, mod_rerank = load_models()
index, chunks = load_index()
groq_client   = Groq(api_key=GROQ_API_KEY)

# ─── RAG helpers ──────────────────────────────────────────────────────────────
def embed_query(query: str):
    inputs = tok_emb(query, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = mod_emb(**inputs)
    return outputs.last_hidden_state[:, -1, :].float().cpu().numpy()

def retrieve(query: str, k: int = 30):
    if index is None or not chunks:
        return []
    emb = embed_query(query)
    _, indices = index.search(emb, k)
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    q_lower = query.lower()
    condition = next((c for c in CONDITION_MAP if c in q_lower), None)
    if not condition:
        return results
    sec = CONDITION_MAP[condition]
    filtered = [
        ch for ch in results
        if str(ch.get("metadata", {}).get("section_number", "")).startswith(sec)
        or "general" in str(ch.get("metadata", {}).get("section_number", "")).lower()
    ]
    return filtered or results

def rerank(query: str, retrieved: list, top_k: int = 3):
    if not retrieved:
        return []
    pairs  = [[query, ch["text"]] for ch in retrieved]
    inputs = tok_rerank(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        scores = mod_rerank(**inputs).logits.view(-1).float()
    ranked = torch.argsort(scores, descending=True).tolist()
    return [retrieved[i] for i in ranked[:top_k]]

def build_context(chunks_list: list) -> str:
    ctx = ""
    for i, ch in enumerate(chunks_list):
        meta = ch.get("metadata", {})
        name = meta.get("section_name", meta.get("source", "WHO Guidelines"))
        num  = meta.get("section_number", f"Page {meta.get('page', 'N/A')}")
        ctx += f"\n--- Evidence {i+1} ---\nSource: {name}, Section {num}\nContent: {ch.get('text','')}\n"
    return ctx

def stream_answer(query: str, top_chunks: list):
    """Generator — yields tokens from Groq. Used with st.write_stream()"""
    if not top_chunks:
        yield "I could not find relevant clinical evidence in the WHO guidelines."
        return
    context     = build_context(top_chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    stream = groq_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=2048,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Care360")
    st.caption("Clinical AI · WHO Rehabilitation Guidelines")
    st.divider()
    st.markdown("**Knowledge base**")
    if index is not None:
        st.success(f"✅ {len(chunks):,} chunks loaded")
    else:
        st.error("❌ FAISS index not found")
    st.divider()
    if st.button("🗑 Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ─── Main UI ──────────────────────────────────────────────────────────────────
st.title("🏥 Care360 — Clinical AI Assistant")
st.caption("Ask any rehabilitation question. Answers are grounded in WHO evidence.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Evidence sources"):
                for src in msg["sources"]:
                    st.markdown(f"""
<div class="source-card">
  <div class="source-title">{src['section_name']} — Section {src['section_number']}</div>
  <div class="source-text">{src['text'][:350]}...</div>
</div>""", unsafe_allow_html=True)

# Input
user_input = st.chat_input("Ask a clinical question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Searching evidence..."):
        retrieved  = retrieve(user_input, k=30)
        top_chunks = rerank(user_input, retrieved, top_k=3)

    with st.chat_message("assistant"):
        answer = st.write_stream(stream_answer(user_input, top_chunks))

        sources = []
        if top_chunks:
            src_html = ""
            for ch in top_chunks:
                meta = ch.get("metadata", {})
                name = meta.get("section_name", "Unknown")
                num  = meta.get("section_number", "Unknown")
                text = ch.get("text", "")
                sources.append({"section_name": name, "section_number": num, "text": text})
                src_html += f"""
<div class="source-card">
  <div class="source-title">{name} — Section {num}</div>
  <div class="source-text">{text[:350]}...</div>
</div>"""
            with st.expander("📚 Evidence sources"):
                st.markdown(src_html, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
