import streamlit as st
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, os, json, hashlib, time, glob

st.set_page_config(page_title="ئێجیۆکەیت (Chomani)", page_icon="📚")
st.title("📚💬 چاتبۆتی ئێجیۆکەیت")
st.caption("تایبەت بە خوێندنی خۆڕایی و سکۆڵەرشیپ")

CLAUDE_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
client = Anthropic(api_key=CLAUDE_API_KEY)

KNOWLEDGE_DIR = "knowledge"
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMB_DIM = 384
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
TOP_K = 10
CHUNK_SIZE = 600
OVERLAP = 200

def hash_docs(doc_paths):
    h = hashlib.sha256()
    for p in sorted(doc_paths):
        h.update(p.encode())
        try:
            h.update(str(os.path.getmtime(p)).encode())
            h.update(str(os.path.getsize(p)).encode())
        except Exception:
            pass
    return h.hexdigest()

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words): break
        start = max(0, end - overlap)
    return chunks

def build_or_load_index():
    paths = glob.glob(os.path.join(KNOWLEDGE_DIR, "*.txt"))
    if not paths:
        st.warning("⚠️ هیچ پەڕگەی .txt نەدۆزرایەوە لە پەڕگەی 'knowledge/' — تکایە پەڕگە زیاد بکە.")
        return None, None

    doc_hash = hash_docs(paths)
    meta_path = "index_meta.json"
    index_path = "index.faiss"

    if os.path.exists(index_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("doc_hash") == doc_hash:
                index = faiss.read_index(index_path)
                return index, meta["items"]
        except Exception:
            pass

    st.info("🏗️ دروستکردنی ئیندەکس ... تکایە چاوەرێکە")
    emb = SentenceTransformer(EMB_MODEL_NAME)
    items = []
    for p in paths:
        txt = load_text(p)
        for ch in chunk_text(txt):
            items.append({"text": ch, "source": os.path.basename(p)})

    texts = [i["text"] for i in items]
    vecs = emb.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(EMB_DIM)
    index.add(vecs)
    faiss.write_index(index, index_path)

    meta = {"doc_hash": doc_hash, "items": items}
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    st.success("✅ ئیندەکس دروستکرا")
    return index, items

def retrieve(index, items, query, k=TOP_K):
    emb = SentenceTransformer(EMB_MODEL_NAME)
    q_vec = emb.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_vec, k)
    return [items[i] for i in I[0] if 0 <= i < len(items)]

def ask_claude(question, context, model_id=DEFAULT_MODEL, temperature=0.7, max_tokens=2200):
    system_msg = (
        "تۆ یاریدەدەری زانستی و خوێندنیت بە زمانی کوردی (سۆرانی). "
        "وەڵامەکانت دەبێت بە وردەکاری و زانستی بنووسیت. "
        "تەنها لەسەر زانیاریەکانی ناو کۆنتێکست وەڵام بدە. "
        "ئەگەر زانیارییەکە نەبوو، بڵێ 'لە بنکەی زانیاری نەدۆزرایەوە'."
    )
    user_msg = (
        f"### کۆنتێکست:\n{context}\n\n"
        f"### پرسیار:\n{question}\n\n"
        "### ڕێنمایی:\n"
        "- بە زمانی کوردی سۆرانی وەڵام بدە.\n"
        "- بە ڕوونی و بە وردەکاری بنووسە.\n"
    )
    msg = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_msg,
        messages=[{"role": "user", "content": user_msg}],
    )
    return msg.content[0].text

with st.spinner("ئامادەکردنی بنکەی زانیاری..."):
    index, items = build_or_load_index()

st.subheader("💬 پرسیار بکە")
question = st.text_area("پرسیارەکەت بنووسە دەربارەی خوێندن یاخود ڕێنماییەکانی خوێندن:")

if st.button("ناردن / Send", disabled=index is None):
    if not question.strip():
        st.warning("تکایە پرسیار بنووسە.")
    elif index is None:
        st.error("بنکەی زانیاری ئامادە نییە.")
    else:
        with st.spinner("گەڕان لە ناو بنکەی زانیاری..."):
            hits = retrieve(index, items, question)
            context = "\n\n".join([h["text"] for h in hits])
        with st.spinner("وەڵامدانەوە لەلایەن Claude..."):
            try:
                answer = ask_claude(question, context)
                st.markdown("### 🧠 وەڵام:")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ هەڵەیەک ڕووی دا: {e}")
