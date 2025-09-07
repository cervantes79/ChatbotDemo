#!/usr/bin/env python3
"""
Generic ICAR V2 CLI Interface
Universal Domain-Agnostic RAG System
Author: Barış Genç
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from src.generic_agent import GenericICARAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generic_icar.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Load environment variables and check configurations"""
    load_dotenv()
    
    # OpenAI API key is optional for Generic ICAR V2
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        logger.warning("OPENAI_API_KEY not found. Generic ICAR V2 will use LLM-free processing mode.")
    
    weather_key = os.getenv('OPENWEATHER_API_KEY')
    if not weather_key:
        logger.warning("OPENWEATHER_API_KEY not found. Weather functionality will be disabled.")
    
    return True

def create_sample_data():
    """Create sample documents if they don't exist"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Check if we have any documents
    doc_files = list(data_dir.rglob("*.pdf")) + list(data_dir.rglob("*.txt"))
    
    if not doc_files:
        logger.info("No documents found. Creating sample documents...")
        try:
            import subprocess
            result = subprocess.run([sys.executable, "create_sample_pdfs.py"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Sample documents created successfully")
            else:
                logger.warning("Failed to create sample documents automatically")
        except Exception as e:
            logger.warning(f"Error creating sample documents: {e}")

def initialize_generic_icar(processing_mode: str = "hybrid"):
    """Initialize Generic ICAR V2 Agent"""
    logger.info(f"Initializing Generic ICAR V2 Agent (Universal Domain-Agnostic RAG)...")
    logger.info(f"Author: Barış Genç")
    logger.info(f"Processing Mode: {processing_mode}")
    
    agent = GenericICARAgent(processing_mode=processing_mode)
    
    if not agent.initialize():
        logger.error("Failed to initialize Generic ICAR V2 Agent")
        return None
    
    # Load documents with universal processing
    data_dir = "data"
    if os.path.exists(data_dir):
        logger.info("Loading documents with Generic ICAR V2 universal processing...")
        success = agent.load_documents(data_dir)
        if success:
            stats = agent.get_stats()
            logger.info(f"Generic ICAR V2: Processed {stats['documents_stored']} documents")
            logger.info(f"Generic ICAR V2: Created {stats['chunks_stored']} processed chunks")
        else:
            logger.warning("Failed to load documents, but Generic ICAR V2 will still work for basic queries")
    else:
        logger.warning("Data directory not found. Generic ICAR V2 will work without knowledge base.")
    
    return agent

def print_welcome():
    """Print welcome message and Generic ICAR V2 information"""
    print("\n" + "="*80)
    print("🌍 GENERIC ICAR V2 - Universal Domain-Agnostic RAG System")
    print("="*80)
    print("Author: Barış Genç | Revolutionary Generic RAG Methodology")
    print("\nThis universal system works with ANY document type across ALL domains:")
    print("• 📚 Education: Course materials, handbooks, syllabi")
    print("• 🏥 Healthcare: Patient guides, medical protocols, treatment plans")
    print("• 🛒 E-Commerce: Product catalogs, shipping policies, customer service")
    print("• ⚖️ Legal: Contracts, employment law, compliance documents")
    print("• 🍳 ANY Domain: Cooking, sports, technology, arts - EVERYTHING!")
    print("\n🧠 Generic ICAR V2 Features:")
    print("• 🔄 LLM-Free Processing: Cost-effective NLTK + scikit-learn")
    print("• 📊 Smart Chunking: Keywords + Summary + Context reconstruction")
    print("• 🎯 Universal Patterns: Works without domain-specific configuration")
    print("• 💬 Natural Interaction: Intelligent query understanding")
    print("\nType your questions naturally - Generic ICAR V2 will understand!")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("Type 'stats' to see system information.")
    print("Type 'mode' to change processing mode.")
    print("="*80 + "\n")

def print_stats(agent):
    """Print Generic ICAR V2 system statistics"""
    stats = agent.get_stats()
    print("\n" + "-"*60)
    print("🌍 GENERIC ICAR V2 SYSTEM STATISTICS")
    print("-"*60)
    print(f"System: {stats.get('system', 'Generic ICAR V2')}")
    print(f"Author: {stats.get('author', 'Barış Genç')}")
    print(f"Processing Mode: {stats.get('processing_mode', 'hybrid')}")
    print(f"Documents Processed: {stats.get('documents_stored', 0)}")
    print(f"Chunks Created: {stats.get('chunks_stored', 0)}")
    print(f"Vector Collection Size: {stats.get('documents_count', 0)}")
    print(f"Agent Initialized: {stats.get('is_initialized', False)}")
    print(f"Enhanced Vector Store: {'✅' if stats.get('vector_store_ready', False) else '❌'}")
    print(f"Weather API: {'✅' if stats.get('weather_api_ready', False) else '❌'}")
    print(f"LLM Available: {'✅' if stats.get('llm_available', False) else '❌ (LLM-free mode)'}")
    processor_info = stats.get('processor_info', {})
    print(f"NLP Libraries: {'✅' if processor_info.get('sklearn_available', False) else '❌'}")
    print("-"*60 + "\n")

def change_processing_mode(agent):
    """Allow user to change processing mode"""
    print("\n🔧 Available Processing Modes:")
    print("1. keywords - Fast TF-IDF based keyword extraction")
    print("2. summary - Comprehensive extractive summarization") 
    print("3. hybrid - Best of both approaches (recommended)")
    
    mode_choice = input("Choose mode (1-3) or press Enter for current: ").strip()
    
    mode_map = {"1": "keywords", "2": "summary", "3": "hybrid"}
    
    if mode_choice in mode_map:
        new_mode = mode_map[mode_choice]
        agent.processing_mode = new_mode
        agent.generic_processor.processing_mode = new_mode
        print(f"✅ Processing mode changed to: {new_mode}")
    else:
        print(f"Keeping current mode: {agent.processing_mode}")

def main():
    """Main CLI interface for Generic ICAR V2"""
    try:
        # Setup environment
        if not setup_environment():
            return 1
        
        # Create sample data if needed
        create_sample_data()
        
        # Initialize Generic ICAR V2 Agent
        agent = initialize_generic_icar(processing_mode="hybrid")
        if not agent:
            logger.error("Failed to initialize Generic ICAR V2 Agent. Exiting.")
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
                    print("\nGeneric ICAR Bot: Goodbye! Thank you for using the Universal Domain-Agnostic RAG system by Barış Genç.")
                    break
                
                # Check for stats command
                if user_input.lower() in ['stats', 'status', 'info']:
                    print_stats(agent)
                    continue
                
                # Check for mode change command
                if user_input.lower() in ['mode', 'processing', 'switch']:
                    change_processing_mode(agent)
                    continue
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Process the query with Generic ICAR V2
                print("\nGeneric ICAR Bot: ", end="")
                result = agent.process_query(user_input)
                
                # Display response
                print(result['response'])
                
                # Display Generic ICAR V2 reasoning
                print(f"\n💭 Generic ICAR Analysis: {result['reasoning']}")
                print(f"🔧 Action Taken: {result['action_taken']}")
                print()
                
            except KeyboardInterrupt:
                print("\n\nGeneric ICAR Bot: Goodbye! Thank you for using the Universal Domain-Agnostic RAG system by Barış Genç.")
                break
            except EOFError:
                print("\n\nGeneric ICAR Bot: Goodbye! Thank you for using the Universal Domain-Agnostic RAG system by Barış Genç.")
                break
            except Exception as e:
                logger.error(f"Error in Generic ICAR main loop: {e}")
                print(f"\nGeneric ICAR Bot: I encountered an error: {e}")
                print("Please try again or type 'quit' to exit.\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error in Generic ICAR V2: {e}")
        print(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())