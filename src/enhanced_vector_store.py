import logging
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import asdict
import chromadb
from chromadb.config import Settings
from src.generic_processor import ProcessedChunk, GenericNLPProcessor

logger = logging.getLogger(__name__)

class EnhancedVectorStore:
    """
    Generic ICAR V2 - Enhanced Vector Store
    Stores processed chunks (keywords/summaries) with document references
    
    Author: Barış Genç
    """
    
    def __init__(self, collection_name: str = "generic_icar_v2", persist_directory: str = "./chroma_db_v2"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.processor = None
        
        # Document storage for reconstruction
        self.document_store = {}  # doc_id -> full document content
        self.chunk_metadata = {}  # chunk_id -> chunk metadata
        
        logger.info(f"Enhanced Vector Store initialized - Collection: {collection_name}")
    
    def initialize(self, processing_mode: str = "hybrid") -> bool:
        """Initialize vector store and NLP processor"""
        try:
            # Initialize ChromaDB
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
                logger.info(f"Using existing collection: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Generic ICAR V2 processed chunks"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
            # Initialize generic processor
            self.processor = GenericNLPProcessor(processing_mode=processing_mode)
            
            # Load existing document store
            self._load_document_store()
            
            logger.info("Enhanced Vector Store initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Vector Store: {str(e)}")
            return False
    
    def add_document(self, text: str, doc_id: str, source: str = "", metadata: Dict[str, Any] = None) -> bool:
        """Add document to vector store using generic processing"""
        try:
            if not self.processor or not self.collection:
                logger.error("Vector store not initialized")
                return False
            
            # Store full document for reconstruction
            self.document_store[doc_id] = {
                "text": text,
                "source": source,
                "metadata": metadata or {}
            }
            
            # Process document into chunks
            chunks = self.processor.process_document(text, doc_id, source)
            
            if not chunks:
                logger.warning(f"No chunks generated for document {doc_id}")
                return False
            
            # Prepare data for ChromaDB
            chunk_ids = []
            chunk_contents = []
            chunk_metadatas = []
            
            for chunk in chunks:
                chunk_ids.append(chunk.chunk_id)
                chunk_contents.append(chunk.content)
                
                # Store chunk metadata for reconstruction
                self.chunk_metadata[chunk.chunk_id] = chunk.to_dict()
                
                # Metadata for ChromaDB
                chunk_metadata = {
                    "doc_id": chunk.doc_id,
                    "chunk_type": chunk.chunk_type,
                    "section": chunk.section or "",
                    "position": chunk.position,
                    "confidence": chunk.confidence,
                    "source": source,
                    "processing_mode": self.processor.processing_mode
                }
                
                # Add custom metadata
                if metadata:
                    chunk_metadata.update(metadata)
                
                chunk_metadatas.append(chunk_metadata)
            
            # Add to ChromaDB
            self.collection.add(
                ids=chunk_ids,
                documents=chunk_contents,
                metadatas=chunk_metadatas
            )
            
            # Save document store
            self._save_document_store()
            
            logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add multiple documents"""
        try:
            success_count = 0
            
            for doc in documents:
                text = doc.get("text", "")
                doc_id = doc.get("doc_id", f"doc_{success_count}")
                source = doc.get("source", "")
                metadata = doc.get("metadata", {})
                
                if self.add_document(text, doc_id, source, metadata):
                    success_count += 1
            
            logger.info(f"Successfully added {success_count}/{len(documents)} documents")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 5, filter_metadata: Dict[str, Any] = None) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Search for similar chunks and reconstruct document context
        
        Returns:
            List of (reconstructed_text, metadata, similarity_score)
        """
        try:
            if not self.processor or not self.collection:
                logger.error("Vector store not initialized")
                return []
            
            # Process query using same methodology
            query_chunk = self.processor.process_query(query)
            if not query_chunk:
                logger.error("Failed to process query")
                return []
            
            # Search in ChromaDB
            search_results = self.collection.query(
                query_texts=[query_chunk.content],
                n_results=k,
                where=filter_metadata if filter_metadata else None,
                include=["documents", "metadatas", "distances"]
            )
            
            if not search_results['ids'][0]:
                return []
            
            # Reconstruct context from document references
            results = []
            for i, chunk_id in enumerate(search_results['ids'][0]):
                try:
                    # Get chunk metadata
                    chunk_meta = self.chunk_metadata.get(chunk_id, {})
                    doc_id = chunk_meta.get('doc_id')
                    
                    if not doc_id or doc_id not in self.document_store:
                        continue
                    
                    # Reconstruct context
                    reconstructed_text = self._reconstruct_context(chunk_id, doc_id)
                    
                    # Calculate similarity score (ChromaDB returns distances, convert to similarity)
                    distance = search_results['distances'][0][i]
                    similarity_score = max(0, 1 - distance)  # Convert distance to similarity
                    
                    # Enhanced metadata
                    metadata = {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "source": self.document_store[doc_id].get("source", ""),
                        "chunk_type": chunk_meta.get("chunk_type", ""),
                        "confidence": chunk_meta.get("confidence", 0.0),
                        "position": chunk_meta.get("position", 0),
                        "original_metadata": self.document_store[doc_id].get("metadata", {})
                    }
                    
                    results.append((reconstructed_text, metadata, similarity_score))
                    
                except Exception as e:
                    logger.error(f"Error processing result {i}: {str(e)}")
                    continue
            
            # Sort by similarity score
            results.sort(key=lambda x: x[2], reverse=True)
            
            logger.info(f"Found {len(results)} similar chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def _reconstruct_context(self, chunk_id: str, doc_id: str, context_window: int = 1) -> str:
        """
        Reconstruct context around a chunk using document references
        
        Args:
            chunk_id: Target chunk ID
            doc_id: Document ID
            context_window: Number of surrounding chunks to include
        """
        try:
            chunk_meta = self.chunk_metadata.get(chunk_id, {})
            chunk_position = chunk_meta.get("position", 0)
            
            # Get document content
            doc_content = self.document_store[doc_id]["text"]
            
            # Get original chunk text
            chunk_original = chunk_meta.get("original_text", "")
            
            if context_window <= 0:
                return chunk_original
            
            # Find related chunks in the document
            related_chunks = []
            for other_chunk_id, other_meta in self.chunk_metadata.items():
                if (other_meta.get("doc_id") == doc_id and 
                    abs(other_meta.get("position", 0) - chunk_position) <= context_window):
                    related_chunks.append((other_meta.get("position", 0), other_meta.get("original_text", "")))
            
            # Sort by position and reconstruct
            related_chunks.sort(key=lambda x: x[0])
            context_parts = [text for pos, text in related_chunks if text.strip()]
            
            if context_parts:
                return " [...] ".join(context_parts)
            else:
                return chunk_original
            
        except Exception as e:
            logger.error(f"Error reconstructing context: {str(e)}")
            # Fallback to original chunk
            chunk_meta = self.chunk_metadata.get(chunk_id, {})
            return chunk_meta.get("original_text", "")
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get full document by ID"""
        return self.document_store.get(doc_id)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk metadata by ID"""
        return self.chunk_metadata.get(chunk_id)
    
    def _save_document_store(self):
        """Save document store to disk"""
        try:
            store_path = os.path.join(self.persist_directory, "document_store.json")
            os.makedirs(self.persist_directory, exist_ok=True)
            
            with open(store_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "documents": self.document_store,
                    "chunks": self.chunk_metadata
                }, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving document store: {str(e)}")
    
    def _load_document_store(self):
        """Load document store from disk"""
        try:
            store_path = os.path.join(self.persist_directory, "document_store.json")
            
            if os.path.exists(store_path):
                with open(store_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.document_store = data.get("documents", {})
                    self.chunk_metadata = data.get("chunks", {})
                    
                logger.info(f"Loaded {len(self.document_store)} documents and {len(self.chunk_metadata)} chunks")
                
        except Exception as e:
            logger.error(f"Error loading document store: {str(e)}")
    
    def get_collection_count(self) -> int:
        """Get number of chunks in collection"""
        try:
            if self.collection:
                return self.collection.count()
            return 0
        except:
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            stats = {
                "collection_name": self.collection_name,
                "total_chunks": self.get_collection_count(),
                "total_documents": len(self.document_store),
                "processing_mode": self.processor.processing_mode if self.processor else "unknown",
                "version": "Generic ICAR V2",
                "author": "Barış Genç"
            }
            
            if self.processor:
                stats.update(self.processor.get_stats())
                
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"error": str(e)}
    
    def reset_collection(self) -> bool:
        """Reset collection and document store"""
        try:
            if self.client and self.collection:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Generic ICAR V2 processed chunks"}
                )
                
                # Clear stores
                self.document_store = {}
                self.chunk_metadata = {}
                
                # Save empty store
                self._save_document_store()
                
                logger.info("Collection and document store reset successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            return False