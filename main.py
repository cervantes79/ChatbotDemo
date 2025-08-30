#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from src.agent_core import AgenticChatbot

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Load environment variables and check required configurations"""
    load_dotenv()
    
    required_vars = ['OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please create a .env file with your API keys. See .env.example for reference.")
        return False
    
    weather_key = os.getenv('OPENWEATHER_API_KEY')
    if not weather_key:
        logger.warning("OPENWEATHER_API_KEY not found. Weather functionality will be disabled.")
    
    return True

def create_sample_data():
    """Create sample PDF documents if they don't exist"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Check if we have any PDF files
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.info("No PDF files found. Creating sample documents...")
        try:
            # Try to run the PDF creation script
            import subprocess
            result = subprocess.run([sys.executable, "create_sample_pdfs.py"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Sample PDF documents created successfully")
            else:
                logger.warning("Failed to create PDF documents, using text fallback")
        except Exception as e:
            logger.warning(f"Error creating sample documents: {e}")

def initialize_chatbot():
    """Initialize the chatbot and load documents"""
    logger.info("Initializing Agentic RAG Chatbot...")
    
    chatbot = AgenticChatbot()
    
    if not chatbot.initialize():
        logger.error("Failed to initialize chatbot")
        return None
    
    # Load documents
    data_dir = "data"
    if os.path.exists(data_dir):
        logger.info("Loading documents into knowledge base...")
        success = chatbot.load_documents(data_dir)
        if success:
            stats = chatbot.get_stats()
            logger.info(f"Knowledge base loaded with {stats['documents_count']} document chunks")
        else:
            logger.warning("Failed to load documents, but chatbot will still work for basic queries")
    else:
        logger.warning("Data directory not found. Chatbot will work without knowledge base.")
    
    return chatbot

def print_welcome():
    """Print welcome message and instructions"""
    print("\n" + "="*60)
    print("🤖 AGENTIC RAG CHATBOT DEMO")
    print("="*60)
    print("Welcome! This chatbot can:")
    print("• Answer questions using loaded documents")
    print("• Provide weather information for cities")
    print("• Respond to general conversations")
    print("\nThe bot will explain its reasoning for each response.")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("Type 'stats' to see system information.")
    print("="*60 + "\n")

def print_stats(chatbot):
    """Print chatbot statistics"""
    stats = chatbot.get_stats()
    print("\n" + "-"*40)
    print("SYSTEM STATISTICS")
    print("-"*40)
    print(f"Documents loaded: {stats.get('documents_count', 0)}")
    print(f"Chatbot initialized: {stats.get('is_initialized', False)}")
    print(f"Vector store ready: {stats.get('vector_store_ready', False)}")
    print(f"Weather API ready: {stats.get('weather_api_ready', False)}")
    print("-"*40 + "\n")

def main():
    """Main CLI interface"""
    try:
        # Setup environment
        if not setup_environment():
            return 1
        
        # Create sample data if needed
        create_sample_data()
        
        # Initialize chatbot
        chatbot = initialize_chatbot()
        if not chatbot:
            logger.error("Failed to initialize chatbot. Exiting.")
            return 1
        
        # Print welcome message
        print_welcome()
        
        # Main conversation loop
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\nBot: Goodbye! Thank you for using the Agentic RAG Chatbot.")
                    break
                
                # Check for stats command
                if user_input.lower() in ['stats', 'status', 'info']:
                    print_stats(chatbot)
                    continue
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Process the query
                print("\nBot: ", end="")
                result = chatbot.process_query(user_input)
                
                # Display response
                print(result['response'])
                
                # Display reasoning
                print(f"\n💭 Decision Process: {result['reasoning']}")
                print(f"🔧 Action Taken: {result['action_taken']}")
                print()
                
            except KeyboardInterrupt:
                print("\n\nBot: Goodbye! Thank you for using the Agentic RAG Chatbot.")
                break
            except EOFError:
                print("\n\nBot: Goodbye! Thank you for using the Agentic RAG Chatbot.")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print(f"\nBot: I encountered an error: {e}")
                print("Please try again or type 'quit' to exit.\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())