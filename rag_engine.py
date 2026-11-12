import os
import pathlib
import numpy as np
import faiss
import requests
import sentence_transformers

INDEX_PATH = "rag.index"
DOCS_DIR = "data/rag_docs"

class RAGEngine:
    def __init__(self, hf_api_token: str = None):
        self.hf_token = hf_api_token
        # Small 384-d model (only ~90MB memory footprint)
        self.embed_model = sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.chunks = []
        self.sources = []
        self.index = None
        
    def recursive_chunk(self, text, chunk_size=300, chunk_overlap=50):
        """Recursive chunker matching notebook Step 1a"""
        separators = ["\n\n", "\n", ". ", " "]
        def _split(t, seps):
            if len(t) <= chunk_size: return [t]
            if not seps:
                step = chunk_size - chunk_overlap
                return [t[i:i + chunk_size] for i in range(0, len(t), step)]
            sep = seps[0]
            parts = t.split(sep)
            pieces = [p + sep if i < len(parts) - 1 else p for i, p in enumerate(parts)]
            chunks, current = [], ""
            for piece in pieces:
                if len(current) + len(piece) <= chunk_size:
                    current += piece
                else:
                    if current: chunks.append(current)
                    if len(piece) > chunk_size:
                        chunks.extend(_split(piece, seps[1:]))
                        current = ""
                    else: current = piece
            if current: chunks.append(current)
            return chunks

        raw_chunks = [c for c in _split(text, separators) if c.strip()]
        out = []
        for i, c in enumerate(raw_chunks):
            if i < len(raw_chunks) - 1:
                out.append(c + raw_chunks[i + 1][:chunk_overlap])
            else:
                out.append(c)
        return out

    def build_or_load_index(self):
        """Builds index from scratch or loads rag.index if present"""
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            # Re-read text corpus mapping
            self.load_corpus()
            return f"Loaded index from disk with {self.index.ntotal} vectors."
        
        self.load_corpus()
        if not self.chunks:
            return "No text files found in data/rag_docs."

        embeddings = self.embed_model.encode(self.chunks, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(384)
        self.index.add(embeddings)
        faiss.write_index(self.index, INDEX_PATH)
        return f"Successfully built FAISS index with {self.index.ntotal} vectors."

    def load_corpus(self):
        self.chunks, self.sources = [], []
        corpus_path = pathlib.Path(DOCS_DIR)
        if not corpus_path.exists():
            os.makedirs(DOCS_DIR, exist_ok=True)
            return

        for p in sorted(corpus_path.glob('*.txt')):
            text = p.read_text(encoding='utf-8')
            for c in self.recursive_chunk(text):
                self.chunks.append(c)
                self.sources.append(p.name)

    def retrieve(self, question: str, k: int = 3):
        """Returns top-k matching chunks"""
        q_vec = self.embed_model.encode([question]).astype('float32')
        faiss.normalize_L2(q_vec)
        scores, ids = self.index.search(q_vec, k)
        
        hits = []
        for i in range(k):
            idx = int(ids[0][i])
            score = float(scores[0][i])
            hits.append({
                'score': score,
                'source': self.sources[idx],
                'chunk': self.chunks[idx]
            })
        return hits

    def generate_answer(self, question: str, k: int = 3):
        """Grounding & augmented prompt construction"""
        hits = self.retrieve(question, k=k)
        
        # Check low similarity score threshold (Rule of thumb: < 0.4 score)[cite: 1]
        top_score = hits[0]['score'] if hits else 0.0
        
        context = ""
        for i, hit in enumerate(hits, start=1):
            context += f"\n[Source {i}: {hit['source']}]\n{hit['chunk']}\n"

        prompt = (
            'Answer the question using ONLY the context below. '
            'If the answer is not in the context, say "I do not know". '
            'Do not guess. Cite the source file for each fact you use.\n\n'
            f'Context:{context}\nQuestion: {question}\nAnswer:'
        )

        if top_score < 0.35:
            return {
                "answer": "I do not know based on the provided course material.",
                "sources": hits,
                "top_score": top_score
            }

        # API Call to DeepSeek / HuggingFace Inference to save local RAM
        answer = self.call_llm_api(prompt)
        return {
            "answer": answer,
            "sources": hits,
            "top_score": top_score
        }

    def call_llm_api(self, prompt: str):
        if not self.hf_token:
            return "API key missing. Set HF_TOKEN environment variable."
        
        # Serverless API call to DeepSeek-R1 / Qwen model
        API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 300, "temperature": 0.2}
        }
        
        try:
            res = requests.post(API_URL, headers=headers, json=payload, timeout=15)
            data = res.json()
            if isinstance(data, list) and 'generated_text' in data[0]:
                return data[0]['generated_text'].replace(prompt, '').strip()
            return data.get('generated_text', str(data))
        except Exception as e:
            return f"Error querying LLM API: {str(e)}"