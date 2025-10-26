# vector_demo.py
import os
import json
import numpy as np
import faiss  # <-- replaced hnswlib with faiss
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify

DATA_DIR = "docs"   # put plain .txt files here
INDEX_PATH = "faiss_index.bin"
DOC_META_PATH = "doc_meta.json"
MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast model

# Load sentence transformer model
model = SentenceTransformer(MODEL_NAME)

# ----------------------------
# Load documents from folder
# ----------------------------
def load_docs(directory=DATA_DIR):
    docs = []
    for fname in os.listdir(directory):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(directory, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        docs.append({"id": fname, "text": text})
    return docs

# ----------------------------
# Build FAISS index
# ----------------------------
def build_index(docs):
    embeddings = np.array(model.encode([d["text"] for d in docs], convert_to_numpy=True), dtype="float32")
    dim = embeddings.shape[1]

    # Create FAISS index (cosine similarity uses inner product with normalized vectors)
    faiss.normalize_L2(embeddings)  # normalize vectors for cosine similarity
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine after normalization
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, INDEX_PATH)

    # Save document metadata
    meta = {str(i): docs[i] for i in range(len(docs))}
    with open(DOC_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# ----------------------------
# Load FAISS index
# ----------------------------
def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOC_META_PATH):
        return None, None
    index = faiss.read_index(INDEX_PATH)
    with open(DOC_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

# ----------------------------
# Build or load index
# ----------------------------
docs = load_docs()
if not os.path.exists(INDEX_PATH) or not os.path.exists(DOC_META_PATH):
    print("Building FAISS index from docs...")
    build_index(docs)
else:
    print("Index and metadata exist. Loading...")

index, meta = load_index()

# ----------------------------
# Flask API to query
# ----------------------------
app = Flask("vector_demo")

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    q = data.get("query", "")
    k = int(data.get("k", 5))
    if not q:
        return jsonify({"error": "query required"}), 400

    # Encode query and normalize for cosine similarity
    q_emb = np.array(model.encode([q], convert_to_numpy=True), dtype="float32")
    faiss.normalize_L2(q_emb)

    # Search in FAISS
    D, I = index.search(q_emb, k)
    results = []
    for lbl, dist in zip(I[0], D[0]):
        doc = meta[str(int(lbl))]
        results.append({"id": doc["id"], "text": doc["text"][:500], "score": float(dist)})

    return jsonify({"query": q, "results": results})

if __name__ == "__main__":
    app.run(port=8000, debug=True)
