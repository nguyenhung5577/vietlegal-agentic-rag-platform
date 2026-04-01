import os
import re
import json
import warnings
import chromadb
import cohere
from typing import List, Dict, Optional

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


# ── Retriever ──────────────────────────────────────────────
class LegalRetriever:
    def __init__(self):
        # 1. Vector index
        self.db = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.chroma_collection = self.db.get_or_create_collection(
            "viet_legal_rag",
            metadata={"hnsw:space": "ip"},
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL_NAME)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
        )

        # 2. BM25 index
        print("Loading BM25 index with Vietnamese tokenization...")
        with open(config.BM25_INDEX_PATH, "r", encoding="utf-8") as f:
            node_data = json.load(f)

        self.nodes = [
            TextNode(text=n["text"], metadata=n["metadata"], id_=n["id"])
            for n in node_data
        ]

        # BM25 nodes: prepend article title so structural terms (e.g. "PHỤ LỤC I") are searchable
        bm25_nodes = []
        for n in node_data:
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

        # 3. Cohere Reranker (API — no local model load)
        print(f"Initializing Cohere reranker: {config.COHERE_RERANK_MODEL}...")
        self.cohere_client = cohere.ClientV2(api_key=config.COHERE_API_KEY)

    def _rerank(self, query: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Rerank nodes via Cohere API and return top-N with updated scores."""
        if not nodes:
            return []

        docs = [n.node.text for n in nodes]
        response = self.cohere_client.rerank(
            model=config.COHERE_RERANK_MODEL,
            query=query,
            documents=docs,
            top_n=config.TOP_K_RERANK,
        )

        reranked = []
        for r in response.results:
            node = nodes[r.index]
            node.score = r.relevance_score
            reranked.append(node)
        return reranked

    def retrieve(self, query: str, expand_links: bool = True) -> List[Dict]:
        """Hybrid retrieval → Cohere rerank → article aggregation → link expansion."""
        query_bundle = QueryBundle(query)

        # Stage 1: Hybrid retrieval (Vector + BM25)
        initial_nodes = self.hybrid_retriever.retrieve(query_bundle)

        # Stage 2: Cohere rerank
        print(f"Reranking {len(initial_nodes)} results via Cohere API...")
        reranked_nodes = self._rerank(query, initial_nodes)

        # Stage 3: Aggregate by article ID
        articles: Dict[str, Dict] = {}
        for node_with_score in reranked_nodes:
            base_id = node_with_score.node.metadata.get("article_id")
            if not base_id:
                base_id = node_with_score.node.id_.split("_C")[0]

            if base_id not in articles:
                full_chunks = self._find_nodes_by_base_id(base_id)
                full_chunks.sort(
                    key=lambda x: int(x.id_.split("_C")[-1]) if "_C" in x.id_ else 0
                )

                cleaned = []
                for c in full_chunks:
                    parts = c.text.split("\n", 1)
                    raw = parts[1].strip() if len(parts) > 1 else c.text.strip()
                    labeled = re.sub(r"^(\d+)\.", r"[KHOẢN \1]", raw, flags=re.MULTILINE)
                    cleaned.append(labeled)

                articles[base_id] = {
                    "id": base_id,
                    "metadata": node_with_score.node.metadata,
                    "text": "\n".join(cleaned),
                    "score": node_with_score.score,
                    "is_reference": False,
                }

        # Stage 4: Cross-reference link expansion
        if expand_links:
            expanded: Dict[str, Dict] = {}
            for base_id, art in articles.items():
                refs_raw = art["metadata"].get("cross_references", "")
                refs = (
                    [r for r in refs_raw.split(";") if r and r != "CURRENT"]
                    if isinstance(refs_raw, str) else []
                )
                for ref_id in refs:
                    if ref_id not in articles and ref_id not in expanded:
                        ref_chunks = self._find_nodes_by_base_id(ref_id)
                        if ref_chunks:
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

        return sorted(articles.values(), key=lambda x: x["score"], reverse=True)

    def _find_nodes_by_base_id(self, base_id: str) -> List[TextNode]:
        matches = []
        for n in self.nodes:
            node_article_id = n.metadata.get("article_id")
            if node_article_id == base_id:
                matches.append(n)
            elif n.id_ == base_id or n.id_.startswith(f"{base_id}_C"):
                matches.append(n)
        return matches


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
