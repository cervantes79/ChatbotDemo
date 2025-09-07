import logging
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
import re
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
from src.enhanced_vector_store import EnhancedVectorStore
from src.external_apis import WeatherAPI

logger = logging.getLogger(__name__)

class GenericActionType(Enum):
    DIRECT_RESPONSE = "direct_response"
    GENERIC_SEARCH = "generic_search"
    WEATHER_API = "weather_api"

class GenericICARAgent:
    """
    Generic ICAR V2 - Universal Domain-Agnostic Agent
    Works with any type of documents using LLM-free processing
    
    Author: Barış Genç
    """
    
    def __init__(self, processing_mode: str = "hybrid", collection_name: str = "generic_icar"):
        self.vector_store = EnhancedVectorStore(collection_name=collection_name)
        self.weather_api = WeatherAPI()
        self.processing_mode = processing_mode
        
        if LANGCHAIN_AVAILABLE:
            try:
                self.llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
            except Exception:
                self.llm = None
                logger.warning("OpenAI API key not found, Generic ICAR will have limited functionality")
        else:
            self.llm = None
            logger.info("LangChain not available - Generic ICAR V2 working in LLM-free mode")
        
        self.is_initialized = False
        
        logger.info(f"Generic ICAR Agent initialized - Mode: {processing_mode}")
    
    def initialize(self) -> bool:
        """Initialize the generic agent"""
        try:
            if not self.vector_store.initialize(processing_mode=self.processing_mode):
                logger.error("Failed to initialize enhanced vector store")
                return False
            
            self.is_initialized = True
            logger.info("Generic ICAR Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Generic ICAR Agent: {str(e)}")
            return False
    
    def decide_action(self, user_query: str) -> Tuple[GenericActionType, str]:
        """
        Generic decision making - domain agnostic
        Uses simple pattern matching for universal applicability
        """
        try:
            query_lower = user_query.lower().strip()
            
            # Weather detection - universal pattern
            weather_keywords = ['weather', 'temperature', 'forecast', 'rain', 'sunny', 'cloudy', 'climate']
            city_patterns = [
                r'weather.*in\s+([a-zA-Z\s]+)',
                r'temperature.*in\s+([a-zA-Z\s]+)', 
                r'forecast.*for\s+([a-zA-Z\s]+)',
                r'how.*weather.*([a-zA-Z\s]+)'
            ]
            
            if any(keyword in query_lower for keyword in weather_keywords):
                for pattern in city_patterns:
                    match = re.search(pattern, query_lower)
                    if match:
                        city = match.group(1).strip()
                        if len(city) > 1:
                            reasoning = f"Generic ICAR: Weather query detected for '{city}' using universal pattern matching."
                            return GenericActionType.WEATHER_API, reasoning
                
                reasoning = "Generic ICAR: Weather intent detected but no city specified. Will handle via weather API."
                return GenericActionType.WEATHER_API, reasoning
            
            # Simple greeting detection - universal
            greeting_patterns = [
                'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
                'how are you', 'what are you', 'who are you', 'thanks', 'thank you',
                'bye', 'goodbye', 'help', 'what can you do', 'greetings'
            ]
            
            if any(pattern in query_lower for pattern in greeting_patterns) or len(query_lower) < 15:
                reasoning = "Generic ICAR: Universal greeting/simple query pattern detected. No domain-specific search needed."
                return GenericActionType.DIRECT_RESPONSE, reasoning
            
            # Everything else goes to generic search
            reasoning = f"Generic ICAR: Complex query detected. Using domain-agnostic search with {self.processing_mode} processing."
            return GenericActionType.GENERIC_SEARCH, reasoning
            
        except Exception as e:
            logger.error(f"Error in Generic ICAR decision making: {str(e)}")
            return GenericActionType.DIRECT_RESPONSE, "Generic ICAR decision engine encountered an error."
    
    def execute_direct_response(self, user_query: str) -> str:
        """Execute direct response using LLM or fallback"""
        try:
            if self.llm is None:
                return f"Hello! This is Generic ICAR V2 by Barış Genç. I can help with document search and weather queries, but I need an OpenAI API key for advanced conversational features. You asked: '{user_query}'"
            
            if not LANGCHAIN_AVAILABLE:
                return f"Hello! This is Generic ICAR V2 by Barış Genç. I can help with document search and weather queries, but I need LangChain for advanced conversational features. You asked: '{user_query}'"
            
            messages = [
                SystemMessage(content="You are a helpful Generic ICAR V2 assistant by Barış Genç. Provide concise and friendly responses. You work with any type of document using domain-agnostic processing."),
                HumanMessage(content=user_query)
            ]
            
            response = self.llm(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error in Generic ICAR direct response: {str(e)}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again."
    
    def execute_generic_search(self, user_query: str) -> str:
        """
        Execute generic search using enhanced vector store
        Works with any document type/domain
        """
        try:
            if self.vector_store.get_collection_count() == 0:
                return "I don't have any documents in my knowledge base yet. Please load some documents first using the Generic ICAR V2 system."
            
            # Search using enhanced vector store
            search_results = self.vector_store.similarity_search(user_query, k=3)
            
            if not search_results:
                return "I couldn't find any relevant information for your question. The Generic ICAR V2 system searched through all available documents but found no matches. Could you try rephrasing your question?"
            
            # Reconstruct context from results
            contexts = []
            sources = []
            for reconstructed_text, metadata, similarity_score in search_results:
                contexts.append(reconstructed_text)
                source = metadata.get('source', 'Unknown')
                doc_id = metadata.get('doc_id', 'Unknown')
                chunk_type = metadata.get('chunk_type', 'unknown')
                sources.append(f"{source} ({chunk_type}, score: {similarity_score:.2f})")
            
            combined_context = "\n\n---\n\n".join(contexts)
            
            if self.llm is None:
                return f"Generic ICAR V2 found relevant information:\n\n{combined_context}\n\nSources: {', '.join(sources)}\n\n(Note: OpenAI API key needed for enhanced response generation)"
            
            # Generate response using LLM
            system_prompt = f"""You are Generic ICAR V2 by Barış Genç, a domain-agnostic document assistant.
Answer the question based on the provided context from the enhanced vector store.
The context was retrieved using {self.processing_mode} processing mode.
Use only the information from the context to answer the question.
If the context doesn't contain enough information, say so politely.
Provide clear and helpful answers."""
            
            user_prompt = f"""Context (retrieved via Generic ICAR V2):
{combined_context}

Sources: {', '.join(sources)}

Question: {user_query}

Please answer based on the provided context."""
            
            if not LANGCHAIN_AVAILABLE:
                return f"Generic ICAR V2 found relevant information:\n\n{combined_context}\n\nSources: {', '.join(sources)}\n\n(Note: LangChain needed for enhanced response generation)"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error in Generic ICAR search execution: {str(e)}")
            return "I encountered an error while searching through the knowledge base. Please try again."
    
    def execute_weather_api(self, user_query: str) -> str:
        """Execute weather API call - same as before"""
        try:
            query_lower = user_query.lower()
            
            # Extract city name
            city_patterns = [
                r'weather.*in\s+([a-zA-Z\s]+)',
                r'temperature.*in\s+([a-zA-Z\s]+)',
                r'forecast.*for\s+([a-zA-Z\s]+)',
                r'how.*weather.*([a-zA-Z\s]+)'
            ]
            
            city = None
            for pattern in city_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    city = match.group(1).strip()
                    break
            
            if not city:
                return "I'd be happy to help you with weather information! Could you please specify which city you'd like to know about? For example: 'What's the weather in Tokyo?'"
            
            weather_data = self.weather_api.get_weather(city)
            
            if weather_data:
                return self.weather_api.format_weather_response(weather_data)
            else:
                return f"I couldn't retrieve weather information for '{city}'. Please check the city name and try again."
                
        except Exception as e:
            logger.error(f"Error in weather API execution: {str(e)}")
            return "I encountered an error while fetching weather information. Please try again."
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query using Generic ICAR methodology"""
        try:
            if not self.is_initialized:
                return {
                    "response": "Generic ICAR Agent is not properly initialized. Please check the configuration.",
                    "action_taken": "error", 
                    "reasoning": "Initialization failed"
                }
            
            if not user_query or not user_query.strip():
                return {
                    "response": "Please provide a question or message.",
                    "action_taken": "error",
                    "reasoning": "Empty query received"
                }
            
            # Decide action using generic methodology
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
                response = "I'm not sure how to handle that request using Generic ICAR methodology."
                action_name = "Generic ICAR Unknown Action"
            
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
    
    def load_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Load documents using Generic ICAR V2 processing
        
        Args:
            documents: List of dicts with 'text', 'doc_id', 'source', 'metadata'
        """
        try:
            success = self.vector_store.add_documents(documents)
            
            if success:
                stats = self.get_stats()
                logger.info(f"Generic ICAR: Loaded {len(documents)} documents")
                logger.info(f"Generic ICAR: Total chunks in system: {stats.get('total_chunks', 0)}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading documents with Generic ICAR: {str(e)}")
            return False
    
    def load_document_from_file(self, file_path: str, doc_id: str = None) -> bool:
        """Load single document from file"""
        try:
            import os
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not doc_id:
                doc_id = os.path.splitext(os.path.basename(file_path))[0]
            
            documents = [{
                "text": text,
                "doc_id": doc_id,
                "source": file_path,
                "metadata": {"file_path": file_path}
            }]
            
            return self.load_documents(documents)
            
        except Exception as e:
            logger.error(f"Error loading document from file {file_path}: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Generic ICAR system statistics"""
        try:
            vector_stats = self.vector_store.get_stats()
            
            return {
                "system": "Generic ICAR V2",
                "author": "Barış Genç", 
                "processing_mode": self.processing_mode,
                "is_initialized": self.is_initialized,
                "llm_available": self.llm is not None,
                "weather_api_ready": self.weather_api.api_key is not None,
                **vector_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting Generic ICAR stats: {str(e)}")
            return {"error": str(e)}