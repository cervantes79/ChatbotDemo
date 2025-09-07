#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from src.agent_core import ICARChatbot

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
    """Initialize the ICAR chatbot and load documents with concept extraction"""
    logger.info("Initializing ICAR Chatbot (Intelligent Concept-Aware RAG)...")
    logger.info("Author: Barış Genç")
    
    chatbot = ICARChatbot()
    
    if not chatbot.initialize():
        logger.error("Failed to initialize ICAR chatbot")
        return None
    
    # Load documents with concept extraction
    data_dir = "data"
    if os.path.exists(data_dir):
        logger.info("Loading documents with ICAR concept extraction...")
        success = chatbot.load_documents(data_dir)
        if success:
            stats = chatbot.get_stats()
            logger.info(f"ICAR: Knowledge base loaded with {stats['documents_count']} document chunks")
            logger.info(f"ICAR: Extracted {stats.get('concepts_extracted', 0)} concepts")
        else:
            logger.warning("Failed to load documents, but ICAR chatbot will still work for basic queries")
    else:
        logger.warning("Data directory not found. ICAR will work without knowledge base.")
    
    return chatbot

def print_welcome():
    """Print welcome message and ICAR system information"""
    print("\n" + "="*70)
    print("ICAR CHATBOT DEMO - Intelligent Concept-Aware RAG")
    print("="*70)
    print("Author: Barış Genç | Methodology: ICAR")
    print("\nThis advanced chatbot uses ICAR methodology and can:")
    print("• Understand your intent through concept extraction")
    print("• Answer questions using intelligent concept-based retrieval")
    print("• Provide weather information for cities")
    print("• Engage in natural conversations")
    print("• Show detailed reasoning behind each decision")
    print("\nThe ICAR system will explain its concept analysis for each response.")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("Type 'stats' to see ICAR system information.")
    print("="*70 + "\n")

def print_stats(chatbot):
    """Print ICAR system statistics"""
    stats = chatbot.get_stats()
    print("\n" + "-"*50)
    print("ICAR SYSTEM STATISTICS")
    print("-"*50)
    print(f"System: {stats.get('icar_system', 'ICAR')}")
    print(f"Methodology: {stats.get('methodology', 'By Barış Genç')}")
    print(f"Documents loaded: {stats.get('documents_count', 0)}")
    print(f"Concepts extracted: {stats.get('concepts_extracted', 0)}")
    print(f"Concept index size: {stats.get('concept_index_size', 0)}")
    print(f"ICAR initialized: {stats.get('is_initialized', False)}")
    print(f"Vector store ready: {stats.get('vector_store_ready', False)}")
    print(f"Weather API ready: {stats.get('weather_api_ready', False)}")
    print("-"*50 + "\n")

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
                    print("\nICAR Bot: Goodbye! Thank you for using the ICAR Chatbot by Barış Genç.")
                    break
                
                # Check for stats command
                if user_input.lower() in ['stats', 'status', 'info']:
                    print_stats(chatbot)
                    continue
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Process the query with ICAR
                print("\nICAR Bot: ", end="")
                result = chatbot.process_query(user_input)
                
                # Display response
                print(result['response'])
                
                # Display ICAR reasoning
                print(f"\nICAR Analysis: {result['reasoning']}")
                print(f"Action Taken: {result['action_taken']}")
                print()
                
            except KeyboardInterrupt:
                print("\n\nICAR Bot: Goodbye! Thank you for using the ICAR Chatbot by Barış Genç.")
                break
            except EOFError:
                print("\n\nICAR Bot: Goodbye! Thank you for using the ICAR Chatbot by Barış Genç.")
                break
            except Exception as e:
                logger.error(f"Error in ICAR main loop: {e}")
                print(f"\nICAR Bot: I encountered an error: {e}")
                print("Please try again or type 'quit' to exit.\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())