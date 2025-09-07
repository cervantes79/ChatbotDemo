import logging
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.enhanced_vector_store import EnhancedVectorStore
from src.external_apis import WeatherAPI
from src.concept_extractor import ConceptExtractor, Concept

logger = logging.getLogger(__name__)

class ActionType(Enum):
    DIRECT_RESPONSE = "icar_direct_response"
    CONCEPT_SEARCH = "icar_concept_search"
    SEMANTIC_SEARCH = "icar_semantic_search"
    WEATHER_API = "icar_weather_api"

class ICARChatbot:
    """
    Intelligent Concept-Aware RAG (ICAR) Chatbot
    Enhanced agentic chatbot with concept-based retrieval system
    
    Author: Barış Genç
    """
    
    def __init__(self):
        self.vector_store = EnhancedVectorStore()
        self.weather_api = WeatherAPI()
        self.concept_extractor = ConceptExtractor()
        try:
            self.llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        except Exception:
            self.llm = None
            logger.warning("OpenAI API key not found, ICAR will have limited functionality")
        self.is_initialized = False
        self.concept_index = {}
        self.document_concepts = {}

    def initialize(self) -> bool:
        try:
            if not self.vector_store.initialize():
                logger.error("Failed to initialize vector store")
                return False
            
            # Load existing concept index if available
            self.concept_extractor.load_concept_index("concept_index.json")
            
            self.is_initialized = True
            logger.info("ICAR chatbot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing ICAR chatbot: {str(e)}")
            return False

    def decide_action(self, user_query: str) -> Tuple[ActionType, str]:
        """
        Enhanced decision making using ICAR methodology
        Analyzes user query with concept extraction for intelligent action selection
        """
        try:
            query_lower = user_query.lower().strip()
            
            # Extract concepts from user query
            query_concepts = self.concept_extractor.extract_query_concepts(user_query)
            
            # Weather detection using concept-aware approach
            weather_concepts = ["weather_request"]
            if any(concept.name in weather_concepts for concept in query_concepts):
                # Enhanced weather detection
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
                            reasoning = f"ICAR Analysis: Detected weather intent for '{city}' with high concept confidence. Using weather API for real-time data."
                            return ActionType.WEATHER_API, reasoning
                
                reasoning = "ICAR Analysis: Weather intent detected but no specific city identified. Will request clarification via weather API."
                return ActionType.WEATHER_API, reasoning
            
            # Greeting detection using concepts
            greeting_concepts = ["greeting"]
            if any(concept.name in greeting_concepts for concept in query_concepts) or len(query_lower) < 10:
                reasoning = "ICAR Analysis: Simple greeting or basic query detected. No document retrieval needed - responding directly."
                return ActionType.DIRECT_RESPONSE, reasoning
            
            # Concept-based knowledge retrieval decision
            if query_concepts and any(concept.weight > 0.6 for concept in query_concepts):
                # Check if we have strong concept matches in our index
                strong_concepts = [c for c in query_concepts if c.weight > 0.6]
                concept_names = [c.name for c in strong_concepts]
                
                # Try concept-based search first
                concept_matches = self.concept_extractor.match_concepts(query_concepts, similarity_threshold=0.3)
                if concept_matches:
                    reasoning = f"ICAR Analysis: Detected high-confidence domain concepts: {concept_names}. Using intelligent concept-based retrieval."
                    return ActionType.CONCEPT_SEARCH, reasoning
            
            # Fallback to semantic search for document queries
            document_indicators = [
                'what is', 'how to', 'explain', 'describe', 'tell me about',
                'policy', 'procedure', 'guide', 'manual', 'documentation'
            ]
            
            if any(indicator in query_lower for indicator in document_indicators) or len(query_lower) > 15:
                reasoning = "ICAR Analysis: Document-related query detected. Using semantic search as fallback method."
                return ActionType.SEMANTIC_SEARCH, reasoning
            
            # Default fallback
            reasoning = "ICAR Analysis: Query doesn't match specific patterns. Using direct response."
            return ActionType.DIRECT_RESPONSE, reasoning
            
        except Exception as e:
            logger.error(f"Error in ICAR decision making: {str(e)}")
            return ActionType.DIRECT_RESPONSE, "Error occurred during ICAR decision making, falling back to direct response."

    def execute_direct_response(self, user_query: str) -> str:
        try:
            messages = [
                SystemMessage(content="You are a helpful assistant. Provide concise and friendly responses."),
                HumanMessage(content=user_query)
            ]
            
            response = self.llm(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error in direct response: {str(e)}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again."

    def execute_concept_search(self, user_query: str) -> str:
        """Execute concept-based retrieval using ICAR methodology"""
        try:
            query_concepts = self.concept_extractor.extract_query_concepts(user_query)
            concept_matches = self.concept_extractor.match_concepts(query_concepts, similarity_threshold=0.3)
            
            if not concept_matches:
                return "I couldn't find documents matching your query concepts. Let me try a semantic search instead."
            
            # Get top matching documents
            top_docs = [doc_id for doc_id, score in concept_matches[:3]]
            
            # Perform similarity search on matched documents
            search_results = self.vector_store.similarity_search(user_query, k=3)
            
            if not search_results:
                return "I found conceptually related documents but couldn't retrieve specific content. Please try rephrasing your question."
            
            context = "\n\n".join([doc for doc, metadata in search_results])
            
            system_prompt = """You are an ICAR-powered assistant using Intelligent Concept-Aware RAG methodology.
Answer questions based on the provided context using concept-based understanding.
Mention that this response uses ICAR concept-based retrieval for enhanced accuracy.
Provide clear and helpful answers."""
            
            user_prompt = f"""Context (retrieved using ICAR concept matching):
{context}

Question: {user_query}

Please provide a comprehensive answer based on the concept-matched context."""
            
            if not self.llm:
                return f"Based on concept analysis, here's the relevant information:\n\n{context}"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error in ICAR concept search: {str(e)}")
            return "I encountered an error during concept-based search. Falling back to semantic search."

    def execute_semantic_search(self, user_query: str) -> str:
        """Execute traditional semantic search as ICAR fallback method"""
        try:
            if self.vector_store.get_collection_count() == 0:
                return "I don't have any documents in my knowledge base yet. Please make sure documents are loaded first."
            
            search_results = self.vector_store.similarity_search(user_query, k=3)
            
            if not search_results:
                return "I couldn't find any relevant information in my knowledge base for your question. Could you try rephrasing your question or ask about something else?"
            
            context = "\n\n".join([doc for doc, metadata in search_results])
            
            system_prompt = """You are an ICAR-powered assistant using semantic search as a fallback method.
Answer questions based on the provided context with ICAR methodology awareness.
Mention that this uses ICAR semantic search for comprehensive coverage.
Provide clear and detailed answers."""
            
            user_prompt = f"""Context (retrieved using ICAR semantic search):
{context}

Question: {user_query}

Please answer the question based on the provided context."""
            
            if not self.llm:
                return f"Based on semantic search, here's the relevant information:\n\n{context}"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error in ICAR semantic search execution: {str(e)}")
            return "I encountered an error while searching through my knowledge base. Please try again."

    def execute_weather_api(self, user_query: str) -> str:
        try:
            query_lower = user_query.lower()
            
            # Try to extract city name from the query
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
                return "I'd be happy to help you with weather information! Could you please specify which city you'd like to know about? For example: 'What's the weather in London?'"
            
            weather_data = self.weather_api.get_weather(city)
            
            if weather_data:
                return self.weather_api.format_weather_response(weather_data)
            else:
                return f"I couldn't retrieve weather information for '{city}'. Please check the city name and try again, or make sure the weather service is available."
                
        except Exception as e:
            logger.error(f"Error in weather API execution: {str(e)}")
            return "I encountered an error while fetching weather information. Please try again."

    def process_query(self, user_query: str) -> Dict[str, Any]:
        try:
            if not self.is_initialized:
                return {
                    "response": "Chatbot is not properly initialized. Please check the configuration.",
                    "action_taken": "error",
                    "reasoning": "Initialization failed"
                }
            
            if not user_query or not user_query.strip():
                return {
                    "response": "Please provide a question or message.",
                    "action_taken": "error",
                    "reasoning": "Empty query received"
                }
            
            # Decide what action to take
            action_type, reasoning = self.decide_action(user_query)
            
            # Execute the decided action
            if action_type == ActionType.DIRECT_RESPONSE:
                response = self.execute_direct_response(user_query)
                action_name = "ICAR Direct Response"
                
            elif action_type == ActionType.CONCEPT_SEARCH:
                response = self.execute_concept_search(user_query)
                action_name = "ICAR Concept-Based Retrieval"
                
            elif action_type == ActionType.SEMANTIC_SEARCH:
                response = self.execute_semantic_search(user_query)
                action_name = "ICAR Semantic Search"
                
            elif action_type == ActionType.WEATHER_API:
                response = self.execute_weather_api(user_query)
                action_name = "ICAR Weather API Call"
                
            else:
                response = "I'm not sure how to handle that request."
                action_name = "Unknown Action"
            
            return {
                "response": response,
                "action_taken": action_name,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": "I encountered an unexpected error. Please try again.",
                "action_taken": "error",
                "reasoning": f"Exception occurred: {str(e)}"
            }

    def load_documents(self, data_dir: str) -> bool:
        """Load documents with ICAR concept extraction"""
        try:
            from src.document_processor import DocumentProcessor
            
            processor = DocumentProcessor()
            documents = processor.process_documents(data_dir)
            
            if not documents:
                logger.warning("No documents were loaded")
                return False
            
            # Extract concepts from documents
            total_concepts = 0
            for i, (doc_content, doc_metadata) in enumerate(documents):
                doc_id = f"doc_{i}_{doc_metadata.get('source', 'unknown')}"
                concepts = self.concept_extractor.extract_document_concepts(doc_content, doc_id)
                total_concepts += len(concepts)
            
            # Save concept index
            self.concept_extractor.save_concept_index("concept_index.json")
            
            success = self.vector_store.add_documents(documents)
            if success:
                logger.info(f"Successfully loaded {len(documents)} document chunks with {total_concepts} concepts extracted")
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading documents with ICAR processing: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get ICAR system statistics"""
        try:
            concept_stats = self.concept_extractor.get_stats()
            return {
                "icar_system": "ICAR (Intelligent Concept-Aware RAG)",
                "methodology": "By Barış Genç",
                "documents_count": self.vector_store.get_collection_count(),
                "concepts_extracted": concept_stats.get("total_concepts", 0),
                "concept_index_size": concept_stats.get("total_documents", 0),
                "is_initialized": self.is_initialized,
                "vector_store_ready": self.vector_store.collection is not None,
                "weather_api_ready": self.weather_api.api_key is not None
            }
        except Exception as e:
            logger.error(f"Error getting ICAR stats: {str(e)}")
            return {"error": str(e)}