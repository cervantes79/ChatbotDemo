"""
Traditional RAG Implementation for Benchmark Comparison
Author: Barış Genç
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
import uuid

logger = logging.getLogger(__name__)

class TraditionalRAG:
    """
    Traditional RAG implementation without concept awareness
    Pure semantic similarity-based retrieval for comparison
    """
    
    def __init__(self, persist_directory: str = "traditional_rag_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.collection_name = "traditional_rag_docs"
        
    def initialize(self) -> bool:
        """Initialize traditional RAG system"""
        try:
            # Create persist directory
            Path(self.persist_directory).mkdir(exist_ok=True)
            
            # Initialize Chroma client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing Traditional RAG collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new Traditional RAG collection: {self.collection_name}")
            
            logger.info("Traditional RAG initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Traditional RAG: {e}")
            return False
    
    def add_document(self, text: str, metadata: Dict[str, Any], doc_id: Optional[str] = None) -> str:
        """Add document with simple chunking (no concept processing)"""
        try:
            if not doc_id:
                doc_id = str(uuid.uuid4())[:8]
            
            # Simple chunking without concept awareness
            chunks = self._simple_chunk_document(text)
            
            # Add chunks to vector store
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []
            
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i:03d}"
                
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk_text)
                chunk_metadatas.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "position": i,
                    "source": metadata.get("source", "unknown"),
                    "type": "traditional_chunk"
                })
            
            # Add to vector collection
            if chunk_ids:
                self.collection.add(
                    ids=chunk_ids,
                    documents=chunk_texts,
                    metadatas=chunk_metadatas
                )
                
                logger.info(f"Traditional RAG: Added document {doc_id} with {len(chunk_ids)} chunks")
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document to Traditional RAG: {e}")
            return None
    
    def _simple_chunk_document(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Simple text chunking without intelligent boundaries"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Simple chunking - no sentence boundary detection
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks
    
    def search(self, query: str, k: int = 3) -> List[Tuple[str, Dict[str, Any]]]:
        """Traditional similarity search without concept awareness"""
        try:
            if not self.collection:
                logger.error("Traditional RAG collection not initialized")
                return []
            
            # Pure semantic similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['documents'][0]:
                return []
            
            # Return results without concept processing
            search_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                enhanced_metadata = {
                    **metadata,
                    "distance": distance,
                    "retrieval_method": "traditional_semantic",
                    "rank": i + 1
                }
                
                search_results.append((doc, enhanced_metadata))
            
            logger.debug(f"Traditional RAG returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in Traditional RAG search: {e}")
            return []
    
    def get_collection_count(self) -> int:
        """Get number of documents in collection"""
        try:
            if self.collection:
                return self.collection.count()
            return 0
        except Exception as e:
            logger.error(f"Error getting Traditional RAG collection count: {e}")
            return 0
    
    def reset_collection(self) -> bool:
        """Reset the collection"""
        try:
            if self.client and self.collection:
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            
            logger.info("Traditional RAG collection reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting Traditional RAG collection: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Traditional RAG statistics"""
        try:
            return {
                "collection_count": self.get_collection_count(),
                "persist_directory": self.persist_directory,
                "collection_name": self.collection_name,
                "retrieval_method": "Traditional Semantic Similarity",
                "system_type": "Baseline Traditional RAG"
            }
        except Exception as e:
            logger.error(f"Error getting Traditional RAG stats: {e}")
            return {"error": str(e)}