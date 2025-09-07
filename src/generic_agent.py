"""
Generic ICAR V2 Agent - Universal Domain-Agnostic Implementation
Author: Barış Genç
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.enhanced_vector_store import EnhancedVectorStore
from src.external_apis import WeatherAPI
from src.generic_processor import GenericProcessor

logger = logging.getLogger(__name__)

class GenericActionType(Enum):
    DIRECT_RESPONSE = "generic_icar_direct_response"
    GENERIC_SEARCH = "generic_icar_search"
    WEATHER_API = "generic_icar_weather_api"

class GenericICARAgent:
    """
    Generic ICAR V2 Agent - Universal Domain-Agnostic RAG
    Works with ANY document type across ALL domains
    Author: Barış Genç
    """
    
    def __init__(self, processing_mode: str = "hybrid"):
        self.enhanced_vector_store = EnhancedVectorStore()
        self.weather_api = WeatherAPI()
        self.generic_processor = GenericProcessor(processing_mode=processing_mode)
        self.processing_mode = processing_mode
        
        try:
            self.llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        except Exception:
            self.llm = None
            logger.warning("OpenAI API key not found, Generic ICAR will have limited functionality")
        
        self.is_initialized = False
        self.document_store = {}
    
    def initialize(self) -> bool:
        """Initialize the Generic ICAR V2 agent"""
        try:
            if not self.enhanced_vector_store.initialize():
                logger.error("Failed to initialize enhanced vector store")
                return False
            
            self.is_initialized = True
            logger.info("Generic ICAR V2 Agent initialized successfully")
            logger.info(f"Processing mode: {self.processing_mode}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Generic ICAR Agent: {str(e)}")
            return False
    
    def decide_action(self, user_query: str) -> Tuple[GenericActionType, str]:
        """
        Universal decision making - works for any domain
        Simple, reliable pattern matching
        """
        try:
            query_lower = user_query.lower().strip()
            
            # Universal weather detection
            weather_patterns = [
                r'weather.*in\s+([a-zA-Z\s]+)',
                r'what.*weather.*([a-zA-Z\s]+)',
                r'temperature.*in\s+([a-zA-Z\s]+)',
                r'forecast.*([a-zA-Z\s]+)'
            ]
            
            for pattern in weather_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    city = match.group(1).strip()
                    if city and len(city) > 1:
                        reasoning = f"Generic ICAR Analysis: Detected weather query for '{city}' using universal pattern matching."
                        return GenericActionType.WEATHER_API, reasoning
            
            # Universal greeting detection
            greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "thanks", "thank you", "bye", "goodbye"]
            if any(greeting in query_lower for greeting in greetings) or len(query_lower) < 10:
                reasoning = "Generic ICAR Analysis: Universal greeting pattern detected. No domain-specific search needed."
                return GenericActionType.DIRECT_RESPONSE, reasoning
            
            # Default to generic search for everything else
            reasoning = "Generic ICAR Analysis: Complex query detected. Using domain-agnostic search with " + self.processing_mode + " processing."
            return GenericActionType.GENERIC_SEARCH, reasoning
            
        except Exception as e:
            logger.error(f"Error in Generic ICAR decision making: {str(e)}")
            return GenericActionType.DIRECT_RESPONSE, "Error occurred during decision making."
    
    def execute_direct_response(self, user_query: str) -> str:
        """Execute direct response for greetings and simple queries"""
        try:
            if not self.llm:
                return "Hello! This is Generic ICAR V2 by Barış Genç. I can help with document search across any domain using LLM-free processing. How can I assist you?"
            
            messages = [
                SystemMessage(content="You are Generic ICAR V2 by Barış Genç. Provide friendly, concise responses. Mention that you use Universal Domain-Agnostic RAG methodology."),
                HumanMessage(content=user_query)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error in direct response: {str(e)}")
            return "Hello! This is Generic ICAR V2 by Barış Genç. I can help with document search across any domain."
    
    def execute_generic_search(self, user_query: str) -> str:
        """Execute domain-agnostic search using enhanced processing"""
        try:
            if self.enhanced_vector_store.get_collection_count() == 0:
                return "I don't have any documents loaded yet. Please add documents from any domain - education, healthcare, legal, e-commerce, etc."
            
            # Perform enhanced similarity search
            search_results = self.enhanced_vector_store.similarity_search(user_query, k=3)
            
            if not search_results:
                return "I couldn't find relevant information for your query. Please try rephrasing or add more documents to my knowledge base."
            
            # Use generic processor for context reconstruction
            context_parts = []
            for doc, metadata in search_results:
                chunk_id = metadata.get('chunk_id')
                if chunk_id:
                    # Enhanced context with reconstruction
                    context_parts.append(f"[Document: {metadata.get('source', 'unknown')}]\n{doc}")
                else:
                    context_parts.append(doc)
            
            context = "\n\n".join(context_parts)
            
            if not self.llm:
                return f"Based on Generic ICAR V2 analysis, here's the relevant information:\n\n{context}"
            
            system_prompt = f"""You are Generic ICAR V2 by Barış Genç - a Universal Domain-Agnostic RAG system.
Answer questions based on the provided context using {self.processing_mode} processing.
Work with ANY domain - education, healthcare, legal, e-commerce, etc.
Mention that this uses Generic ICAR V2 methodology for universal document processing."""
            
            user_prompt = f"""Context (retrieved using Generic ICAR V2 {self.processing_mode} processing):
{context}

Question: {user_query}

Please provide a comprehensive answer based on the domain-agnostic context."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error in generic search execution: {str(e)}")
            return "I encountered an error during generic search. Please try again."
    
    def execute_weather_api(self, user_query: str) -> str:
        """Execute weather API with universal city detection"""
        try:
            query_lower = user_query.lower()
            
            # Universal weather patterns
            weather_patterns = [
                r'weather.*in\s+([a-zA-Z\s]+)',
                r'what.*weather.*([a-zA-Z\s]+)',
                r'temperature.*in\s+([a-zA-Z\s]+)',
                r'forecast.*([a-zA-Z\s]+)'
            ]
            
            city = None
            for pattern in weather_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    city = match.group(1).strip()
                    break
            
            if not city:
                return "I'd be happy to help with weather information! Please specify a city. For example: 'What's the weather in Tokyo?'"
            
            weather_data = self.weather_api.get_weather(city)
            
            if weather_data:
                return self.weather_api.format_weather_response(weather_data)
            else:
                return f"I couldn't retrieve weather information for '{city}'. Please check the city name and try again."
                
        except Exception as e:
            logger.error(f"Error in weather API execution: {str(e)}")
            return "I encountered an error while fetching weather information. Please try again."
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process query using Generic ICAR V2 methodology"""
        try:
            if not self.is_initialized:
                return {
                    "response": "Generic ICAR V2 Agent is not properly initialized. Please check the configuration.",
                    "action_taken": "error",
                    "reasoning": "Initialization failed"
                }
            
            if not user_query or not user_query.strip():
                return {
                    "response": "Please provide a question or message.",
                    "action_taken": "error",
                    "reasoning": "Empty query received"
                }
            
            # Decide action using universal patterns
            action_type, reasoning = self.decide_action(user_query)
            
            # Execute the decided action
            if action_type == GenericActionType.DIRECT_RESPONSE:
                response = self.execute_direct_response(user_query)
                action_name = "Generic ICAR Direct Response"
                
            elif action_type == GenericActionType.GENERIC_SEARCH:
                response = self.execute_generic_search(user_query)
                action_name = f"Generic ICAR Search ({self.processing_mode})"
                
            elif action_type == GenericActionType.WEATHER_API:
                response = self.execute_weather_api(user_query)
                action_name = "Generic ICAR Weather API"
                
            else:
                response = "I'm not sure how to handle that request."
                action_name = "Unknown Action"
            
            return {
                "response": response,
                "action_taken": action_name,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"Error processing query with Generic ICAR: {str(e)}")
            return {
                "response": "I encountered an unexpected error. Please try again.",
                "action_taken": "error",
                "reasoning": f"Exception occurred: {str(e)}"
            }
    
    def load_documents(self, data_dir: str) -> bool:
        """Load documents using generic processing"""
        try:
            from src.document_processor import DocumentProcessor
            
            processor = DocumentProcessor()
            documents = processor.process_documents(data_dir)
            
            if not documents:
                logger.warning("No documents were loaded")
                return False
            
            # Process documents with generic processor
            total_chunks = 0
            for text, metadata in documents:
                doc_id = self.enhanced_vector_store.add_document(text, metadata)
                if doc_id:
                    total_chunks += 1
            
            if total_chunks > 0:
                logger.info(f"Generic ICAR V2: Successfully processed {total_chunks} documents using {self.processing_mode} mode")
                return True
            else:
                logger.error("Failed to process any documents")
                return False
            
        except Exception as e:
            logger.error(f"Error loading documents with Generic ICAR: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Generic ICAR V2 system statistics"""
        try:
            processing_stats = self.generic_processor.get_processing_stats()
            vector_stats = self.enhanced_vector_store.get_stats()
            
            return {
                "system": "Generic ICAR V2 (Universal Domain-Agnostic RAG)",
                "author": "Barış Genç",
                "processing_mode": self.processing_mode,
                "documents_count": vector_stats.get("collection_count", 0),
                "documents_stored": vector_stats.get("documents_stored", 0),
                "chunks_stored": vector_stats.get("chunks_stored", 0),
                "is_initialized": self.is_initialized,
                "vector_store_ready": self.enhanced_vector_store.collection is not None,
                "weather_api_ready": self.weather_api.api_key is not None,
                "llm_available": self.llm is not None,
                "processor_info": processing_stats
            }
        except Exception as e:
            logger.error(f"Error getting Generic ICAR stats: {str(e)}")
            return {"error": str(e)}