import os
import sys
import json
from pathlib import Path
import chromadb

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import BaseRetriever

import torch

from src.ingest import HierarchicalLegalIngestor
import src.config as config

os.environ.setdefault("HF_HOME", str(Path(config.MODEL_CACHE_DIR).parent / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", config.MODEL_CACHE_DIR)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", config.MODEL_CACHE_DIR)

class LegalIndexBuilder:
    def __init__(self, rebuild: bool = False):
        # CPU-friendly HuggingFace embedding setup.
        print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
        
        # Performance tuning for CPU
        num_cores = min(config.MAX_CPU_CORES, os.cpu_count() or config.MAX_CPU_CORES)
        torch.set_num_threads(num_cores)
        try:
            torch.set_num_interop_threads(max(1, num_cores // 2))
        except RuntimeError:
            pass
        print(
            "Optimizing embedding for CPU: "
            f"{num_cores} torch threads, batch size {config.EMBED_BATCH_SIZE}"
        )
        
        device = "cpu"

        self.embed_model = HuggingFaceEmbedding(
            model_name=config.EMBEDDING_MODEL_NAME,
            embed_batch_size=config.EMBED_BATCH_SIZE,
            device=device,
            cache_folder=config.MODEL_CACHE_DIR,
            model_kwargs={"torch_dtype": torch.float32},
        )
        
        # Setup ChromaDB — use inner product (dot product) to match Vietnamese_Embedding similarity function
        self.db = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        if rebuild:
            try:
                self.db.delete_collection("viet_legal_rag")
                print("Deleted existing Chroma collection: viet_legal_rag")
            except Exception:
                print("No existing Chroma collection to delete.")
        self.chroma_collection = self.db.get_or_create_collection(
            "viet_legal_rag",
            metadata={"hnsw:space": "ip"},  # ip = inner product (dot product)
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def build_and_save(self):
        # 1. Ingest nodes
        print("Ingesting legal documents...")
        ingestor = HierarchicalLegalIngestor(config.INPUT_DIR)
        raw_nodes = []
        for filename in os.listdir(config.INPUT_DIR):
            if filename.endswith(".md"):
                raw_nodes.extend(ingestor.parse_file(os.path.join(config.INPUT_DIR, filename)))
        
        # 2. Convert to LlamaIndex TextNodes
        li_nodes = []
        for node in raw_nodes:
            li_node = TextNode(
                text=node.text,
                metadata=node.metadata,
                id_=node.article_id
            )
            li_nodes.append(li_node)
        
        # --- RESUME LOGIC START ---
        # Check existing IDs in ChromaDB to avoid re-embedding
        try:
            # get() defaults to returning IDs
            existing_data = self.chroma_collection.get()
            existing_ids = set(existing_data["ids"])
            print(f"Found {len(existing_ids)} existing nodes in ChromaDB.")
        except Exception as e:
            print(f"No existing collection found or error reading it: {e}")
            existing_ids = set()

        nodes_to_index = [n for n in li_nodes if n.id_ not in existing_ids]

        if not nodes_to_index:
            print("All nodes are already indexed. Skipping embedding phase.")
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store, embed_model=self.embed_model
            )
        else:
            if existing_ids:
                print(f"Resuming indexing: {len(nodes_to_index)} nodes remaining (Skipping {len(existing_ids)} already indexed).")
                self.index = VectorStoreIndex.from_vector_store(
                    self.vector_store, embed_model=self.embed_model
                )
                self.index.insert_nodes(nodes_to_index, show_progress=True)
            else:
                print(f"Starting fresh indexing: {len(nodes_to_index)} nodes...")
                # First time: create index normally
                self.index = VectorStoreIndex(
                    nodes_to_index, 
                    storage_context=self.storage_context, 
                    embed_model=self.embed_model,
                    show_progress=True
                )
        # --- RESUME LOGIC END ---
        
        # 4. Save nodes for BM25
        print(f"Persisting {len(li_nodes)} nodes for BM25 search...")
        node_data = [{"text": n.text, "metadata": n.metadata, "id": n.id_} for n in li_nodes]
        with open(config.BM25_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(node_data, f, ensure_ascii=False, indent=2)
            
        print("Index successfully built and persisted.")

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query_bundle, **kwargs):
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)

        fused = {}
        rrf_k = 60

        for source_weight, nodes in (
            (config.HYBRID_VECTOR_WEIGHT, vector_nodes),
            (config.HYBRID_BM25_WEIGHT, bm25_nodes),
        ):
            for rank, node in enumerate(nodes, start=1):
                node_id = node.node.id_
                if node_id not in fused:
                    fused[node_id] = node
                    fused[node_id].score = 0.0
                fused[node_id].score += source_weight / (rrf_k + rank)

        return sorted(fused.values(), key=lambda n: n.score or 0.0, reverse=True)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build VietLegal vector and BM25 indexes.")
    parser.add_argument("--rebuild", action="store_true", help="Delete the Chroma collection before indexing.")
    args = parser.parse_args()

    builder = LegalIndexBuilder(rebuild=args.rebuild)
    builder.build_and_save()

if __name__ == "__main__":
    main()
