import logging
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.vector_store import VectorStore
from src.external_apis import WeatherAPI
from src.concept_extractor import ConceptExtractor, Concept

logger = logging.getLogger(__name__)

class ActionType(Enum):
    DIRECT_RESPONSE = "direct_response"
    CONCEPT_SEARCH = "concept_search"
    SEMANTIC_SEARCH = "semantic_search"
    WEATHER_API = "weather_api"

class ICARChatbot:
    """
    Intelligent Concept-Aware RAG (ICAR) Chatbot
    Enhanced agentic chatbot with concept-based retrieval system
    
    Author: Barış Genç
    """
    
    def __init__(self):
        self.vector_store = VectorStore()
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
                
                # If we have specific domain concepts, use concept search
                domain_categories = ["employee_policies", "company_info", "products", "procedures"]
                if any(concept.category in domain_categories for concept in strong_concepts):
                    reasoning = f"ICAR Analysis: Detected high-confidence domain concepts: {[c.name for c in strong_concepts]}. Using intelligent concept-based retrieval."
                    return ActionType.CONCEPT_SEARCH, reasoning
                else:
                    reasoning = f"ICAR Analysis: General inquiry with concepts: {[c.name for c in query_concepts]}. Using semantic search as fallback."
                    return ActionType.SEMANTIC_SEARCH, reasoning
            
            # Fallback for queries without clear concepts
            if len(query_lower) > 15:
                reasoning = "ICAR Analysis: Complex query without clear concept mapping. Using semantic search to find relevant information."
                return ActionType.SEMANTIC_SEARCH, reasoning
            
            # Final fallback
            reasoning = "ICAR Analysis: Simple query that doesn't match any specific patterns. Providing direct conversational response."
            return ActionType.DIRECT_RESPONSE, reasoning
            
        except Exception as e:
            logger.error(f"Error in ICAR decision making: {str(e)}")
            return ActionType.DIRECT_RESPONSE, "ICAR decision engine encountered an error, falling back to direct response."

    def execute_direct_response(self, user_query: str) -> str:
        try:
            if self.llm is None:
                return f"Hello! This is the ICAR chatbot by Barış Genç. I can help you with general questions, but I need an OpenAI API key for advanced features. You asked: '{user_query}'"
            
            messages = [
                SystemMessage(content="You are a helpful ICAR-powered assistant by Barış Genç. Provide concise and friendly responses."),
                HumanMessage(content=user_query)
            ]
            
            response = self.llm(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error in ICAR direct response: {str(e)}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again."

    def execute_concept_search(self, user_query: str) -> str:
        """
        ICAR Concept-based search using intelligent concept matching
        """
        try:
            if self.vector_store.get_collection_count() == 0:
                return "I don't have any documents in my knowledge base yet. Please make sure documents are loaded first."
            
            # Extract query concepts
            query_concepts = self.concept_extractor.extract_query_concepts(user_query)
            
            # Get all documents with their concepts
            all_documents = []
            for doc_id, doc_concepts in self.document_concepts.items():
                concept_score = self.concept_extractor.get_concept_score(query_concepts, doc_concepts)
                if concept_score > 0.3:  # Minimum concept threshold
                    all_documents.append((doc_id, concept_score))
            
            # Sort by concept score and get top matches
            all_documents.sort(key=lambda x: x[1], reverse=True)
            top_docs = all_documents[:3]
            
            if not top_docs:
                # Fallback to semantic search
                return self.execute_semantic_search(user_query)
            
            # Get actual document content
            context_parts = []
            for doc_id, score in top_docs:
                # This would need integration with actual document storage
                search_results = self.vector_store.similarity_search(user_query, k=1)
                if search_results:
                    context_parts.append(search_results[0][0])  # Get the document text
            
            context = "\n\n".join(context_parts)
            
            system_prompt = """You are an intelligent assistant using ICAR (Intelligent Concept-Aware RAG) methodology.
Answer the question based on the concept-matched context provided.
The context has been specifically selected based on concept analysis for high relevance.
Provide clear, accurate answers using the matched information."""
            
            user_prompt = f"""Context (selected via concept matching):
{context}

Question: {user_query}

Please provide a comprehensive answer based on the concept-matched context."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error in concept search execution: {str(e)}")
            # Fallback to semantic search
            return self.execute_semantic_search(user_query)

    def execute_semantic_search(self, user_query: str) -> str:
        """
        Enhanced semantic search with ICAR fallback capabilities
        """
        try:
            if self.vector_store.get_collection_count() == 0:
                return "I don't have any documents in my knowledge base yet. Please make sure documents are loaded first."
            
            search_results = self.vector_store.similarity_search(user_query, k=3)
            
            if not search_results:
                return "I couldn't find any relevant information in my knowledge base for your question. Could you try rephrasing your question or ask about something else?"
            
            context = "\n\n".join([doc for doc, metadata in search_results])
            
            system_prompt = """You are an ICAR-powered assistant using semantic search as a fallback method.
Answer questions based on the provided context from vector similarity search.
Use only the information from the context to answer the question.
If the context doesn't contain enough information, suggest alternative approaches politely.
Provide clear and concise answers."""
            
            user_prompt = f"""Context (from semantic similarity search):
{context}

Question: {user_query}

Please answer the question based on the provided context."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error in semantic search execution: {str(e)}")
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
            
            # Execute the decided action using ICAR methodology
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
                response = "I'm not sure how to handle that request using ICAR methodology."
                action_name = "ICAR Unknown Action"
            
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
        """
        Enhanced document loading with ICAR concept extraction
        """
        try:
            from src.document_processor import DocumentProcessor
            
            processor = DocumentProcessor()
            documents = processor.process_documents(data_dir)
            
            if not documents:
                logger.warning("No documents were loaded for ICAR processing")
                return False
            
            # Extract concepts from each document using ICAR methodology
            logger.info("Extracting concepts from documents using ICAR...")
            enhanced_documents = []
            
            for i, doc in enumerate(documents):
                doc_id = f"doc_{i}"
                
                # Extract concepts from document
                concepts = self.concept_extractor.extract_document_concepts(
                    doc.page_content, 
                    source=doc.metadata.get('source', 'unknown')
                )
                
                # Store document concepts for later retrieval
                self.document_concepts[doc_id] = concepts
                
                # Add concept information to metadata
                doc.metadata['concepts'] = [concept.to_dict() for concept in concepts]
                doc.metadata['doc_id'] = doc_id
                doc.metadata['concept_count'] = len(concepts)
                
                enhanced_documents.append(doc)
            
            # Build concept index
            document_data = []
            for doc_id, concepts in self.document_concepts.items():
                document_data.append({
                    "id": doc_id,
                    "concepts": concepts
                })
            
            self.concept_index = self.concept_extractor.build_concept_index(document_data)
            
            # Save concept index for future use
            self.concept_extractor.save_concept_index("concept_index.json")
            
            # Load documents into vector store
            success = self.vector_store.add_documents(enhanced_documents)
            
            if success:
                total_concepts = sum(len(concepts) for concepts in self.document_concepts.values())
                logger.info(f"ICAR: Successfully loaded {len(enhanced_documents)} document chunks with {total_concepts} concepts extracted")
                logger.info(f"ICAR: Built concept index with {len(self.concept_index)} unique concepts")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in ICAR document loading: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        try:
            total_concepts = sum(len(concepts) for concepts in self.document_concepts.values())
            return {
                "documents_count": self.vector_store.get_collection_count(),
                "concepts_extracted": total_concepts,
                "concept_index_size": len(self.concept_index),
                "is_initialized": self.is_initialized,
                "vector_store_ready": self.vector_store.collection is not None,
                "weather_api_ready": self.weather_api.api_key is not None,
                "icar_system": "Intelligent Concept-Aware RAG",
                "methodology": "ICAR by Barış Genç"
            }
        except Exception as e:
            logger.error(f"Error getting ICAR stats: {str(e)}")
            return {"error": str(e)}

# Maintain backward compatibility
AgenticChatbot = ICARChatbot