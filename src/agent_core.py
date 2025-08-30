import logging
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
import re
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from src.vector_store import VectorStore
from src.external_apis import WeatherAPI

logger = logging.getLogger(__name__)

class ActionType(Enum):
    DIRECT_RESPONSE = "direct_response"
    VECTOR_SEARCH = "vector_search"
    WEATHER_API = "weather_api"

class AgenticChatbot:
    def __init__(self):
        self.vector_store = VectorStore()
        self.weather_api = WeatherAPI()
        self.llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
        self.is_initialized = False

    def initialize(self) -> bool:
        try:
            if not self.vector_store.initialize():
                logger.error("Failed to initialize vector store")
                return False
            
            self.is_initialized = True
            logger.info("Agentic chatbot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing chatbot: {str(e)}")
            return False

    def decide_action(self, user_query: str) -> Tuple[ActionType, str]:
        try:
            query_lower = user_query.lower().strip()
            
            # Check for weather-related queries
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
                        reasoning = f"User is asking about weather information for '{city}'. I need to use the weather API to get current weather data."
                        return ActionType.WEATHER_API, reasoning
            
            # Check for direct weather queries without city extraction
            if any(keyword in query_lower for keyword in ['weather', 'temperature', 'forecast', 'rain', 'sunny', 'cloudy']):
                reasoning = "User is asking about weather but I couldn't extract a specific city. I'll need to ask for clarification or try to extract the city from context."
                return ActionType.WEATHER_API, reasoning
            
            # Check for simple greetings and basic conversations
            simple_queries = [
                'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
                'how are you', 'what are you', 'who are you', 'thanks', 'thank you',
                'bye', 'goodbye', 'help', 'what can you do'
            ]
            
            if any(greeting in query_lower for greeting in simple_queries) or len(query_lower) < 10:
                reasoning = "This appears to be a simple greeting or basic question that doesn't require searching through documents or external APIs. I can respond directly."
                return ActionType.DIRECT_RESPONSE, reasoning
            
            # Check for document-specific keywords that suggest knowledge base search
            knowledge_indicators = [
                'policy', 'procedure', 'manual', 'guide', 'documentation',
                'company', 'product', 'service', 'feature', 'specification',
                'return', 'refund', 'warranty', 'contact', 'support',
                'how to', 'what is', 'explain', 'define', 'describe'
            ]
            
            if any(indicator in query_lower for indicator in knowledge_indicators):
                reasoning = f"The query contains keywords that suggest the user needs specific information from our knowledge base. I should search through the documents to find relevant information."
                return ActionType.VECTOR_SEARCH, reasoning
            
            # Default to vector search for most other queries
            if len(query_lower) > 15:
                reasoning = "This appears to be a specific question that might be answered in our knowledge base. I'll search through the documents first."
                return ActionType.VECTOR_SEARCH, reasoning
            
            # Fallback to direct response
            reasoning = "The query doesn't clearly fit into weather or knowledge base categories. I'll provide a direct response."
            return ActionType.DIRECT_RESPONSE, reasoning
            
        except Exception as e:
            logger.error(f"Error in decision making: {str(e)}")
            return ActionType.DIRECT_RESPONSE, "Error occurred during decision making, falling back to direct response."

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

    def execute_vector_search(self, user_query: str) -> str:
        try:
            if self.vector_store.get_collection_count() == 0:
                return "I don't have any documents in my knowledge base yet. Please make sure documents are loaded first."
            
            search_results = self.vector_store.similarity_search(user_query, k=3)
            
            if not search_results:
                return "I couldn't find any relevant information in my knowledge base for your question. Could you try rephrasing your question or ask about something else?"
            
            context = "\n\n".join([doc for doc, metadata in search_results])
            
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer the question.
If the context doesn't contain enough information to answer the question, say so politely.
Provide clear and concise answers."""
            
            user_prompt = f"""Context:
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
            logger.error(f"Error in vector search execution: {str(e)}")
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
                action_name = "Direct Response"
                
            elif action_type == ActionType.VECTOR_SEARCH:
                response = self.execute_vector_search(user_query)
                action_name = "Knowledge Base Search"
                
            elif action_type == ActionType.WEATHER_API:
                response = self.execute_weather_api(user_query)
                action_name = "Weather API Call"
                
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
        try:
            from src.document_processor import DocumentProcessor
            
            processor = DocumentProcessor()
            documents = processor.process_documents(data_dir)
            
            if not documents:
                logger.warning("No documents were loaded")
                return False
            
            success = self.vector_store.add_documents(documents)
            if success:
                logger.info(f"Successfully loaded {len(documents)} document chunks")
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        try:
            return {
                "documents_count": self.vector_store.get_collection_count(),
                "is_initialized": self.is_initialized,
                "vector_store_ready": self.vector_store.collection is not None,
                "weather_api_ready": self.weather_api.api_key is not None
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"error": str(e)}