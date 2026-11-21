from collections import defaultdict
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from .config import settings
from .ingest import Chunk



class HybridIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.embedder = SentenceTransformer(settings.embedding_model)
        self.texts = [c.text for c in chunks]
        tokens = [self._tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokens) if tokens else None
        self.embeddings = self.embedder.encode(self.texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32") if self.texts else np.zeros((0, 384), dtype="float32")
        self.faiss_index = None
        if len(self.embeddings):
            self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.faiss_index.add(self.embeddings)

    @staticmethod
    def _tokenize(text: str):
        import re
        return re.findall(r"\b[\w'-]+\b", text.lower())


    def retrieve(self, query: str, top_k: int = 6, rrf_k: int = 60):
        if not self.chunks:
            return []
        k = min(max(1, top_k), len(self.chunks))
        query_vec = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        sem_scores, sem_ids = self.faiss_index.search(query_vec, min(len(self.chunks), max(k * 3, k)))
        sem_rank = {int(i): rank + 1 for rank, i in enumerate(sem_ids[0]) if i >= 0}
        bm_scores = self.bm25.get_scores(self._tokenize(query))
        bm_ids = np.argsort(bm_scores)[::-1][:min(len(self.chunks), max(k * 3, k))]
        bm_rank = {int(i): rank + 1 for rank, i in enumerate(bm_ids)}
        fused = defaultdict(float)
        for i, rank in sem_rank.items():
            fused[i] += 1 / (rrf_k + rank)
        for i, rank in bm_rank.items():
            fused[i] += 1 / (rrf_k + rank)
        order = sorted(fused, key=fused.get, reverse=True)[:k]
        # Approximate confidence is normalized relative to strongest retrieved fusion score.
        best = max(fused.values()) if fused else 0
        results = []
        for i in order:
            conf = (fused[i] / best) if best else 0
            # Semantic cosine score is also used to avoid a lexical-only false positive.
            cosine = float(np.dot(query_vec[0], self.embeddings[i]))
            results.append({"chunk": self.chunks[i], "score": float(conf), "cosine": cosine, "bm25": float(bm_scores[i])})
        return results
