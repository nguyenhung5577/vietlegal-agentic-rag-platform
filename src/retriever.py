import os
import sys
import re
import json
import warnings
import unicodedata
from pathlib import Path
import chromadb
import cohere
from typing import List, Dict, Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from underthesea import word_tokenize

import src.config as config
from src.index_builder import HybridRetriever

os.environ.setdefault("HF_HOME", str(Path(config.MODEL_CACHE_DIR).parent / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", config.MODEL_CACHE_DIR)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", config.MODEL_CACHE_DIR)


# ── Singleton ──────────────────────────────────────────────
_retriever_instance: Optional["LegalRetriever"] = None

def get_retriever() -> "LegalRetriever":
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = LegalRetriever()
    return _retriever_instance


def _format_results(results: list) -> list[str]:
    formatted = []
    for res in results:
        type_prefix = "KẾT QUẢ CHÍNH" if not res.get("is_reference") else "VĂN BẢN DẪN CHIẾU"
        meta = res["metadata"]
        doc_name = meta.get("title", meta.get("number", ""))
        doc_num = meta.get("number", "")
        header = f"=== {type_prefix}: {doc_name} - {meta.get('article')} ==="
        content = (
            f"{header}\n"
            f"Tên văn bản: {doc_name} (số hiệu: {doc_num})\n"
            f"Tiêu đề điều: {meta.get('article')}\n"
            f"Nội dung luật:\n{res['text']}\n"
            f"====================================\n"
        )
        formatted.append(content)
    return formatted


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFD", value or "")
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", value.lower()).strip()


# ── Retriever ──────────────────────────────────────────────
class LegalRetriever:
    def __init__(self):
        print("Loading persisted legal nodes...")
        with open(config.BM25_INDEX_PATH, "r", encoding="utf-8") as f:
            self.node_data = json.load(f)

        self.nodes = [
            TextNode(text=n["text"], metadata=n["metadata"], id_=n["id"])
            for n in self.node_data
        ]
        self._build_lookup_indexes()

        self.db = None
        self.chroma_collection = None
        self.vector_store = None
        self.embed_model = None
        self.index = None
        self.bm25_retriever = None
        self.vector_retriever = None
        self.hybrid_retriever = None
        self.cohere_client = None

        # BM25 nodes: prepend article title so structural terms (e.g. "PHỤ LỤC I") are searchable
        # 3. Cohere Reranker (API — optional)
    def _ensure_hybrid_retriever(self) -> HybridRetriever:
        if self.hybrid_retriever is not None:
            return self.hybrid_retriever

        print("Loading vector and BM25 retrievers...")
        self.db = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.chroma_collection = self.db.get_or_create_collection(
            "viet_legal_rag",
            metadata={"hnsw:space": "ip"},
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.embed_model = HuggingFaceEmbedding(
            model_name=config.EMBEDDING_MODEL_NAME,
            embed_batch_size=config.EMBED_BATCH_SIZE,
            device="cpu",
            cache_folder=config.MODEL_CACHE_DIR,
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
        )

        bm25_nodes = []
        for n in self.node_data:
            content = n["text"].split("\n", 1)[-1].strip() if "\n" in n["text"] else n["text"]
            article_title = n["metadata"].get("article", "")
            bm25_text = f"{article_title}\n{content}" if article_title else content
            bm25_nodes.append(TextNode(text=bm25_text, metadata=n["metadata"], id_=n["id"]))

        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=bm25_nodes,
            similarity_top_k=config.TOP_K_RETRIEVE,
            tokenizer=word_tokenize,
        )
        self.vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=config.TOP_K_RETRIEVE,
        )
        self.hybrid_retriever = HybridRetriever(self.vector_retriever, self.bm25_retriever)
        return self.hybrid_retriever

    def _ensure_cohere_client(self):
        if self.cohere_client is not None:
            return self.cohere_client
        if config.COHERE_API_KEY:
            print(f"Initializing Cohere reranker: {config.COHERE_RERANK_MODEL}...")
            self.cohere_client = cohere.ClientV2(api_key=config.COHERE_API_KEY)
        else:
            print("COHERE_API_KEY is not set. Using hybrid retrieval scores without API rerank.")
        return self.cohere_client

    def _build_lookup_indexes(self):
        self.article_chunks: Dict[str, List[TextNode]] = {}
        self.article_number_index: Dict[str, List[str]] = {}
        self.article_doc_text: Dict[str, str] = {}
        self.appendix_index: Dict[str, List[str]] = {}

        for node in self.nodes:
            base_id = node.metadata.get("article_id") or node.id_.split("_C")[0]
            self.article_chunks.setdefault(base_id, []).append(node)

        for base_id, chunks in self.article_chunks.items():
            chunks.sort(key=self._chunk_sort_key)
            meta = chunks[0].metadata
            article_number = str(meta.get("article_number") or "")
            if not article_number:
                match = re.search(r"Điều\s+(\d+)", meta.get("article", ""), re.IGNORECASE)
                article_number = match.group(1) if match else ""
            if article_number:
                self.article_number_index.setdefault(article_number, []).append(base_id)

            doc_text = " ".join(
                str(meta.get(key, ""))
                for key in ("title", "number", "source", "authority", "article", "appendix")
            )
            self.article_doc_text[base_id] = _normalize_text(doc_text)

            appendix = meta.get("appendix", "")
            if appendix:
                appendix_key = _normalize_text(f"{meta.get('number', '')} {appendix}")
                self.appendix_index.setdefault(appendix_key, []).append(base_id)

    def _chunk_sort_key(self, node: TextNode) -> int:
        match = re.search(r"_C(\d+)$", node.id_)
        return int(match.group(1)) if match else 0

    def _rerank(self, query: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Rerank nodes via Cohere API and return top-N with updated scores."""
        if not nodes:
            return []
        cohere_client = self._ensure_cohere_client()
        if cohere_client is None:
            return sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)[: config.TOP_K_RERANK]

        docs = [n.node.text for n in nodes]
        try:
            response = cohere_client.rerank(
                model=config.COHERE_RERANK_MODEL,
                query=query,
                documents=docs,
                top_n=config.TOP_K_RERANK,
            )
        except Exception as exc:
            print(f"Cohere rerank failed ({exc}). Falling back to hybrid scores.")
            return sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)[: config.TOP_K_RERANK]

        reranked = []
        for r in response.results:
            node = nodes[r.index]
            node.score = r.relevance_score
            reranked.append(node)
        return reranked

    def retrieve(self, query: str, expand_links: bool = True) -> List[Dict]:
        """Hybrid retrieval → Cohere rerank → article aggregation → link expansion."""
        articles: Dict[str, Dict] = {}

        for base_id in self._resolve_exact_articles(query):
            exact_article = self._article_result_from_base_id(base_id, score=1.25)
            if exact_article:
                exact_article["retrieval_reason"] = "exact_article_lookup"
                articles[base_id] = exact_article

        for base_id in self._resolve_exact_appendices(query):
            appendix_article = self._article_result_from_base_id(base_id, score=1.15)
            if appendix_article:
                appendix_article["retrieval_reason"] = "exact_appendix_lookup"
                articles[base_id] = appendix_article

        if config.EXACT_LOOKUP_SHORT_CIRCUIT and articles and self._is_precise_lookup_query(query):
            if expand_links:
                self._expand_cross_references(articles)
            return sorted(articles.values(), key=lambda x: x["score"], reverse=True)

        query_bundle = QueryBundle(query)

        # Stage 1: Hybrid retrieval (Vector + BM25)
        initial_nodes = self._ensure_hybrid_retriever().retrieve(query_bundle)

        # Stage 2: optional Cohere rerank
        rerank_label = "Cohere API" if config.COHERE_API_KEY else "hybrid-score fallback"
        print(
            f"Reranking {len(initial_nodes)} fused results via {rerank_label} "
            f"for query: {query[:90]}"
        )
        reranked_nodes = self._rerank(query, initial_nodes)

        # Stage 3: Aggregate by article ID
        for node_with_score in reranked_nodes:
            base_id = node_with_score.node.metadata.get("article_id")
            if not base_id:
                base_id = node_with_score.node.id_.split("_C")[0]

            if base_id not in articles:
                article = self._article_result_from_base_id(base_id, score=node_with_score.score or 0.0)
                if article:
                    articles[base_id] = article

        # Stage 4: Cross-reference link expansion
        if expand_links:
            self._expand_cross_references(articles)

        return sorted(articles.values(), key=lambda x: x["score"], reverse=True)

    def _is_precise_lookup_query(self, query: str) -> bool:
        normalized = _normalize_text(query)
        has_exact_target = bool(re.search(r"\bdieu\s+\d+\b", normalized)) or "phu luc" in normalized
        if not has_exact_target:
            return False

        broad_intents = (
            "so sanh",
            "khac nhau",
            "doi chieu",
            "phan tich",
            "tu van",
            "duoc khong",
            "can luu y",
        )
        return not any(intent in normalized for intent in broad_intents)

    def _expand_cross_references(self, articles: Dict[str, Dict]) -> None:
        expanded: Dict[str, Dict] = {}
        for base_id, art in list(articles.items()):
            refs_raw = art["metadata"].get("cross_references", "")
            refs = (
                [r for r in refs_raw.split(";") if r and r != "CURRENT"]
                if isinstance(refs_raw, str) else []
            )
            for ref_id in refs:
                if ref_id in articles or ref_id in expanded:
                    continue
                ref_chunks = self._find_nodes_by_base_id(ref_id)
                if not ref_chunks:
                    continue
                ref_chunks.sort(
                    key=lambda x: int(x.id_.split("_C")[-1]) if "_C" in x.id_ else 0
                )
                ref_texts = []
                for rc in ref_chunks:
                    parts = rc.text.split("\n", 1)
                    ref_texts.append(parts[1].strip() if len(parts) > 1 else rc.text.strip())
                expanded[ref_id] = {
                    "id": ref_id,
                    "metadata": ref_chunks[0].metadata,
                    "text": "\n".join(ref_texts),
                    "score": art["score"] * 0.8,
                    "is_reference": True,
                    "referenced_by": art["metadata"].get("article", base_id),
                }
        articles.update(expanded)

    def _find_nodes_by_base_id(self, base_id: str) -> List[TextNode]:
        return list(self.article_chunks.get(base_id, []))

    def _resolve_exact_articles(self, query: str) -> List[str]:
        normalized = _normalize_text(query)
        match = re.search(r"\bdieu\s+(\d+)\b", normalized)
        if not match:
            return []

        candidates = self.article_number_index.get(match.group(1), [])
        if not candidates:
            return []

        scored = []
        query_tokens = re.findall(r"\d+/\d+/[a-z0-9-]+|\w+", normalized)
        for base_id in candidates:
            doc_text = self.article_doc_text.get(base_id, "")
            score = sum(1 for token in query_tokens if len(token) >= 3 and token in doc_text)
            scored.append((score, base_id))

        scored.sort(reverse=True)
        if scored and scored[0][0] > 0:
            best_score = scored[0][0]
            return [base_id for score, base_id in scored if score == best_score][:3]
        return candidates[:3]

    def _resolve_exact_appendices(self, query: str) -> List[str]:
        normalized = _normalize_text(query)
        wants_appendix = "phu luc" in normalized or "danh muc nganh nghe han che" in normalized
        if not wants_appendix:
            return []

        roman = None
        roman_match = re.search(r"phu luc\s+([ivxlcdm]+)", normalized)
        if roman_match:
            roman = roman_match.group(1).upper()
        elif "danh muc nganh nghe han che" in normalized:
            roman = "I"

        doc_hint = ""
        if "31" in normalized or "31/2021" in normalized or "nghi dinh 31" in normalized:
            doc_hint = "31/2021/nd-cp"

        matches = []
        for appendix_key, base_ids in self.appendix_index.items():
            if roman and f"phu luc {roman.lower()}" not in appendix_key:
                continue
            if doc_hint and doc_hint not in appendix_key:
                continue
            matches.extend(base_ids)

        query_tokens = [
            token
            for token in re.findall(r"\w+", normalized)
            if len(token) >= 4 and token not in {"phu", "luc", "nghi", "dinh", "nha", "dau", "nuoc", "ngoai"}
        ]
        scored = []
        wants_education = "giao duc" in normalized
        for base_id in matches:
            doc_text = self.article_doc_text.get(base_id, "")
            score = sum(1 for token in query_tokens if token in doc_text)
            if wants_education and "giao duc" in doc_text:
                score += 5
            scored.append((score, base_id))

        scored.sort(reverse=True)
        if scored and scored[0][0] > 0:
            return [base_id for score, base_id in scored if score > 0][:5]
        return matches[:5]

    def _article_result_from_base_id(self, base_id: str, score: float) -> Optional[Dict]:
        full_chunks = self._find_nodes_by_base_id(base_id)
        if not full_chunks:
            return None

        cleaned = []
        for chunk in full_chunks:
            parts = chunk.text.split("\n", 1)
            raw = parts[1].strip() if len(parts) > 1 else chunk.text.strip()
            labeled = re.sub(r"^(\d+)\.", r"[KHOẢN \1]", raw, flags=re.MULTILINE)
            cleaned.append(labeled)

        return {
            "id": base_id,
            "metadata": full_chunks[0].metadata,
            "text": "\n".join(cleaned),
            "score": score,
            "is_reference": False,
        }


def main():
    import sys
    retriever = LegalRetriever()
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Điều kiện thành lập doanh nghiệp"
    print(f"Querying: {query}")
    results = retriever.retrieve(query)
    for i, res in enumerate(results):
        type_label = "[PHỤ TRỢ]" if res.get("is_reference") else "[CHÍNH]"
        ref_info = f" (Dẫn chiếu từ {res['referenced_by']})" if res.get("is_reference") else ""
        print(f"\n--- Result {i+1} {type_label} (Score: {res['score']:.4f}){ref_info} ---")
        print(f"ID: {res['id']}")
        print(f"Văn bản: {res['metadata'].get('number')} - {res['metadata'].get('authority')}")
        print(f"Tiêu đề: {res['metadata'].get('article')}")
        print("-" * 20)
        print(res["text"])
        print("-" * 50)


if __name__ == "__main__":
    main()
