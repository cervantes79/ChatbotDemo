"""
Enhanced Vector Store with Document Reconstruction
ICAR methodology enhancement
Author: Barış Genç
"""

import json
import logging
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
import uuid

logger = logging.getLogger(__name__)

class EnhancedVectorStore:
    """
    Enhanced vector store with document reconstruction capabilities
    for Generic ICAR V2 methodology
    Author: Barış Genç
    """
    
    def __init__(self, persist_directory: str = "chroma_db_v2"):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.collection_name = "icar_documents_v2"
        self.document_store_path = f"{persist_directory}/document_store.json"
        self.document_store = {"documents": {}, "chunks": {}}
        
    def initialize(self) -> bool:
        """Initialize the enhanced vector store"""
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
                logger.info(f"Loaded existing ICAR collection: {self.collection_name}")
            except ValueError:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new ICAR collection: {self.collection_name}")
            
            # Load document store
            self._load_document_store()
            
            logger.info("Enhanced vector store initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing enhanced vector store: {e}")
            return False
    
    def _load_document_store(self):
        """Load document store from disk"""
        try:
            if Path(self.document_store_path).exists():
                with open(self.document_store_path, 'r', encoding='utf-8') as f:
                    self.document_store = json.load(f)
                logger.info(f"Loaded document store with {len(self.document_store.get('documents', {}))} documents")
            else:
                self.document_store = {"documents": {}, "chunks": {}}
                logger.info("Created new document store")
                
        except Exception as e:
            logger.error(f"Error loading document store: {e}")
            self.document_store = {"documents": {}, "chunks": {}}
    
    def _save_document_store(self):
        """Save document store to disk"""
        try:
            Path(self.document_store_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.document_store_path, 'w', encoding='utf-8') as f:
                json.dump(self.document_store, f, indent=2, ensure_ascii=False)
            logger.debug("Document store saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving document store: {e}")
    
    def add_document(self, text: str, metadata: Dict[str, Any], doc_id: Optional[str] = None) -> str:
        """Add a single document with reconstruction metadata"""
        try:
            if not doc_id:
                doc_id = str(uuid.uuid4())[:8]
            
            # Store full document
            self.document_store["documents"][doc_id] = {
                "text": text,
                "source": metadata.get("source", "unknown"),
                "metadata": metadata
            }
            
            # Process document into chunks using generic processor
            from src.generic_processor import GenericProcessor
            processor = GenericProcessor(processing_mode="hybrid")
            processed_chunks = processor.process_document(text, doc_id, metadata)
            
            # Add chunks to vector store
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []
            
            for chunk in processed_chunks:
                # Store chunk in document store
                self.document_store["chunks"][chunk.chunk_id] = {
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": chunk.chunk_type,
                    "content": chunk.content,
                    "original_text": chunk.original_text,
                    "doc_id": chunk.doc_id,
                    "section": chunk.section,
                    "position": chunk.position,
                    "confidence": chunk.confidence,
                    "keywords": chunk.keywords,
                    "summary": chunk.summary
                }
                
                # Prepare for vector store
                chunk_ids.append(chunk.chunk_id)
                chunk_texts.append(chunk.content)
                chunk_metadatas.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk.chunk_id,
                    "position": chunk.position,
                    "chunk_type": chunk.chunk_type,
                    "keywords": ",".join(chunk.keywords),
                    "source": metadata.get("source", "unknown")
                })
            
            # Add to vector collection
            if chunk_ids:
                self.collection.add(
                    ids=chunk_ids,
                    documents=chunk_texts,
                    metadatas=chunk_metadatas
                )
                
                logger.info(f"Added document {doc_id} with {len(chunk_ids)} chunks")
            
            # Save document store
            self._save_document_store()
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return None
    
    def add_documents(self, documents: List[Tuple[str, Dict[str, Any]]]) -> bool:
        """Add multiple documents"""
        try:
            added_count = 0
            for text, metadata in documents:
                doc_id = self.add_document(text, metadata)
                if doc_id:
                    added_count += 1
            
            logger.info(f"Successfully added {added_count}/{len(documents)} documents")
            return added_count > 0
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[str, Dict[str, Any]]]:
        """Enhanced similarity search with context reconstruction"""
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return []
            
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['documents'][0]:
                return []
            
            # Reconstruct context from chunks
            search_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                chunk_id = metadata.get('chunk_id')
                
                # Get full chunk info from document store
                if chunk_id in self.document_store["chunks"]:
                    chunk_info = self.document_store["chunks"][chunk_id]
                    
                    # Reconstruct context with original text
                    reconstructed_context = self._reconstruct_chunk_context(
                        chunk_info, query
                    )
                    
                    enhanced_metadata = {
                        **metadata,
                        "distance": distance,
                        "chunk_info": chunk_info,
                        "reconstruction_quality": "high"
                    }
                    
                    search_results.append((reconstructed_context, enhanced_metadata))
                else:
                    # Fallback to basic document
                    search_results.append((doc, {**metadata, "distance": distance}))
            
            logger.debug(f"Similarity search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def _reconstruct_chunk_context(self, chunk_info: Dict, query: str) -> str:
        """Reconstruct context for a chunk with enhanced information"""
        try:
            doc_id = chunk_info.get("doc_id")
            position = chunk_info.get("position", 0)
            
            # Get surrounding chunks for better context
            surrounding_chunks = []
            for chunk_id, chunk_data in self.document_store["chunks"].items():
                if (chunk_data.get("doc_id") == doc_id and 
                    abs(chunk_data.get("position", 0) - position) <= 1):
                    surrounding_chunks.append(chunk_data)
            
            # Sort by position
            surrounding_chunks.sort(key=lambda x: x.get("position", 0))
            
            # Build context
            context_parts = []
            for chunk in surrounding_chunks:
                if chunk.get("position") == position:
                    # Main chunk - use original text with highlighting
                    context_parts.append(f"[MAIN CHUNK]\n{chunk.get('original_text', '')}")
                    if chunk.get('keywords'):
                        context_parts.append(f"[Keywords: {', '.join(chunk.get('keywords', [])[:3])}]")
                else:
                    # Context chunk - use summary
                    context_parts.append(f"[Context]\n{chunk.get('summary', chunk.get('original_text', '')[:200])}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error reconstructing chunk context: {e}")
            return chunk_info.get("original_text", chunk_info.get("content", ""))
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get full document by ID"""
        try:
            return self.document_store["documents"].get(doc_id)
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None
    
    def get_document_chunks(self, doc_id: str) -> List[Dict]:
        """Get all chunks for a document"""
        try:
            chunks = []
            for chunk_id, chunk_data in self.document_store["chunks"].items():
                if chunk_data.get("doc_id") == doc_id:
                    chunks.append(chunk_data)
            
            # Sort by position
            chunks.sort(key=lambda x: x.get("position", 0))
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting document chunks for {doc_id}: {e}")
            return []
    
    def get_collection_count(self) -> int:
        """Get number of documents in collection"""
        try:
            if self.collection:
                return self.collection.count()
            return 0
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0
    
    def reset_collection(self) -> bool:
        """Reset the collection and document store"""
        try:
            if self.client and self.collection:
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            
            # Reset document store
            self.document_store = {"documents": {}, "chunks": {}}
            self._save_document_store()
            
            logger.info("Collection and document store reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced vector store statistics"""
        try:
            return {
                "collection_count": self.get_collection_count(),
                "documents_stored": len(self.document_store.get("documents", {})),
                "chunks_stored": len(self.document_store.get("chunks", {})),
                "persist_directory": self.persist_directory,
                "collection_name": self.collection_name,
                "store_type": "Enhanced Vector Store with Document Reconstruction",
                "methodology": "Generic ICAR V2 by Barış Genç"
            }
        except Exception as e:
            logger.error(f"Error getting enhanced vector store stats: {e}")
            return {"error": str(e)}