import os
import json
import chromadb
from typing import List
from llama_index.core import VectorStoreIndex, StorageContext, Document, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine

import torch

from src.ingest import HierarchicalLegalIngestor
import src.config as config

class LegalIndexBuilder:
    def __init__(self):
        # Initialize BGE-M3 (Standard HuggingFace)
        # Optimized for CPU:
        # 1. Batch size 8 (Standard 32 is too high for CPU cache)
        # 2. bfloat16 (Faster on modern CPUs like Intel/AMD)
        # 3. Explicit thread control
        
        print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
        
        # Performance tuning for CPU
        num_cores = config.MAX_CPU_CORES if hasattr(config, 'MAX_CPU_CORES') else (os.cpu_count() or 8)
        torch.set_num_threads(num_cores)
        print(f"Optimizing for CPU with {num_cores} threads and BF16 (if supported)...")
        
        # Check if BF16 is supported (faster on modern CPUs)
        device = "cpu"
        # BF16 is usually safe and fast on newer CPUs, but let's stick to float32 if unsure
        # or use bfloat16 for a speed boost
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() or hasattr(torch.cpu, 'is_bf16_supported') and torch.cpu.is_bf16_supported() else torch.float32

        self.embed_model = HuggingFaceEmbedding(
            model_name=config.EMBEDDING_MODEL_NAME,
            embed_batch_size=8, # Reduced for CPU stability
            device=device,
            model_kwargs={"torch_dtype": torch_dtype}
        )
        
        # Setup ChromaDB — use inner product (dot product) to match Vietnamese_Embedding similarity function
        self.db = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
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

        # Merge results (simple union with scores for now)
        all_nodes = {}
        for node in vector_nodes:
            all_nodes[node.node.id_] = node
        for node in bm25_nodes:
            if node.node.id_ not in all_nodes:
                all_nodes[node.node.id_] = node
                
        return list(all_nodes.values())

def main():
    builder = LegalIndexBuilder()
    builder.build_and_save()

if __name__ == "__main__":
    main()
