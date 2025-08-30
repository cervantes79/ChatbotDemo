import os
import logging
from typing import List, Optional, Tuple
import chromadb
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection_name: str = "chatbot_documents", persist_directory: str = "chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embeddings = None
        self.client = None
        self.collection = None
        
    def initialize(self) -> bool:
        try:
            if not os.getenv("OPENAI_API_KEY"):
                logger.error("OPENAI_API_KEY not found in environment variables")
                return False
            
            self.embeddings = OpenAIEmbeddings()
            
            os.makedirs(self.persist_directory, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(name=self.collection_name)
                logger.info(f"Created new collection: {self.collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            return False

    def add_documents(self, documents: List[Document]) -> bool:
        try:
            if not self.collection or not self.embeddings:
                logger.error("Vector store not properly initialized")
                return False
            
            if not documents:
                logger.warning("No documents to add")
                return True
            
            logger.info(f"Adding {len(documents)} documents to vector store")
            
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            ids = [f"doc_{i}" for i in range(len(documents))]
            
            embeddings = self.embeddings.embed_documents(texts)
            
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False

    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[str, dict]]:
        try:
            if not self.collection or not self.embeddings:
                logger.error("Vector store not properly initialized")
                return []
            
            query_embedding = self.embeddings.embed_query(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            if not results['documents'] or not results['documents'][0]:
                logger.info("No similar documents found")
                return []
            
            search_results = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'][0] else {}
                search_results.append((doc, metadata))
            
            logger.info(f"Found {len(search_results)} similar documents")
            return search_results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []

    def get_collection_count(self) -> int:
        try:
            if not self.collection:
                return 0
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting collection count: {str(e)}")
            return 0

    def clear_collection(self) -> bool:
        try:
            if not self.client:
                logger.error("Client not initialized")
                return False
            
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
            except:
                logger.info("Collection didn't exist or was already deleted")
            
            self.collection = self.client.create_collection(name=self.collection_name)
            logger.info(f"Created fresh collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False