import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mock_llm import chat

POLICY_DIR = "data/policies"

def load_docs():
    docs = {}
    for fn in sorted(os.listdir(POLICY_DIR)):
        if fn.endswith(".txt"):
            with open(os.path.join(POLICY_DIR, fn), encoding="utf-8") as f:
                docs[fn] = f.read()
    return docs

def chunk_words(text, size=40, overlap=10):
    words = text.split()
    step = size - overlap
    chunks = []
    start = 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return chunks

def build_chunks(size=40, overlap=10):
    out = []
    for fn, text in load_docs().items():
        for c in chunk_words(text, size, overlap):
            out.append({"source": fn, "text": c})
    return out



def build_index(size=40, overlap=10):
    chunks = build_chunks(size, overlap)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    return chunks, vectorizer, matrix

def retrieve(query, k=3, index=None):
    chunks, vectorizer, matrix = index or DEFAULT_INDEX
    scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
    top = scores.argsort()[::-1][:k]
    return [{"source": chunks[i]["source"], "score": float(scores[i]),
             "text": chunks[i]["text"]} for i in top]

DEFAULT_INDEX = build_index()

SYSTEM = (
    "You are a POLICY ASSISTANT. Answer only from the supplied context. "
    "If the answer is not in the context, say you could not find the information."
)

def answer(question, index=None):
    results = retrieve(question, k=3, index=index)
    context = "\n".join(f"[{r['source']}]\n{r['text']}" for r in results)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
    response = chat(messages)
    return response.choices[0].message.content, [r["source"] for r in results]

def evaluate(index=None):
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_hits = fact_hits = fallback_hits = 0
    failures = []
    for _, r in ev.iterrows():
        text, sources = answer(r["question"], index=index)
        if r["expected_doc"] == "NONE":
            if text.startswith("I could not find"):
                fallback_hits += 1
            else:
                failures.append((r["question"], "no fallback", text))
        else:
            d = r["expected_doc"] in sources
            f = r["expected_fact"] in text
            doc_hits += d
            fact_hits += f
            if not (d and f):
                failures.append((r["question"], f"doc={d} fact={f}", text))
    return doc_hits, fact_hits, fallback_hits, failures

if __name__ == "__main__":
    chunks = build_chunks()
    print(len(chunks))                       # 15
    print(pd.Series([c["source"] for c in chunks]).value_counts())

    for r in retrieve("refund policy late delivery"):
        print(r["source"], round(r["score"], 3))

    res = evaluate()
    print(res[:3])
    assert res[:3] == (12, 12, 2)

    res0 = evaluate(index=build_index(40, 0))
    print(res0[:3])
    assert len(res0[3]) == 2


