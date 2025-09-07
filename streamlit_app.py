import streamlit as st
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from src.agent_core import ICARChatbot

# Page configuration
st.set_page_config(
    page_title="ICAR Chatbot Demo - Intelligent Concept-Aware RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def initialize_chatbot():
    """Initialize the ICAR chatbot (cached for performance)"""
    load_dotenv()
    
    # Check required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found. Please set it in your .env file.")
        return None
    
    chatbot = ICARChatbot()
    
    if not chatbot.initialize():
        st.error("Failed to initialize ICAR chatbot")
        return None
    
    # Load documents with ICAR concept extraction
    data_dir = "data"
    if os.path.exists(data_dir):
        with st.spinner("Loading documents with ICAR concept extraction..."):
            success = chatbot.load_documents(data_dir)
            if success:
                stats = chatbot.get_stats()
                st.success(f"ICAR: Knowledge base loaded with {stats['documents_count']} document chunks")
                st.info(f"ICAR: Extracted {stats.get('concepts_extracted', 0)} concepts")
            else:
                st.warning("Failed to load documents, but ICAR chatbot will still work for basic queries")
    
    return chatbot

def create_sample_data():
    """Create sample PDF documents if they don't exist"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        try:
            import subprocess
            import sys
            result = subprocess.run([sys.executable, "create_sample_pdfs.py"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Sample PDF documents created successfully")
                st.rerun()
        except Exception as e:
            st.warning(f"Could not create sample documents: {e}")

def main():
    # Title and header
    st.title("🧠 ICAR Chatbot Demo - Intelligent Concept-Aware RAG")
    st.markdown("**Author**: Barış Genç | **Methodology**: ICAR (Intelligent Concept-Aware RAG)")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🧠 ICAR System Information")
        
        # Initialize chatbot
        chatbot = initialize_chatbot()
        
        if chatbot:
            stats = chatbot.get_stats()
            st.success("✅ ICAR System Initialized")
            st.info(f"📄 Documents: {stats.get('documents_count', 0)}")
            st.info(f"🧠 Concepts: {stats.get('concepts_extracted', 0)}")
            st.info(f"📊 Index Size: {stats.get('concept_index_size', 0)}")
            st.info(f"🗄️ Vector Store: {'✅' if stats.get('vector_store_ready', False) else '❌'}")
            st.info(f"🌤️ Weather API: {'✅' if stats.get('weather_api_ready', False) else '❌'}")
            st.info(f"👨‍💻 Author: {stats.get('methodology', 'Barış Genç')}")
        else:
            st.error("❌ ICAR System Not Initialized")
            st.stop()
        
        st.markdown("---")
        st.header("🧠 ICAR Capabilities")
        st.markdown("""
        This ICAR-powered chatbot can:
        - 🎯 **Concept Analysis**: Understand user intent through concept extraction
        - 📚 **Intelligent Retrieval**: Answer questions using concept-based matching
        - 🔍 **Smart Search**: Fall back to semantic search when needed  
        - 🌤️ **Weather Information**: Get current weather with enhanced detection
        - 💬 **Natural Conversation**: Respond to greetings with concept awareness
        
        The ICAR system explains its concept analysis and decision-making process for each response.
        """)
        
        # Sample data creation
        st.markdown("---")
        if st.button("🔄 Create Sample Documents"):
            create_sample_data()
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🧠 ICAR Chat Interface")
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
                        if "reasoning" in message and "action" in message:
                            with st.expander("🧠 ICAR Analysis Process"):
                                st.write(f"**ICAR Reasoning:** {message['reasoning']}")
                                st.write(f"**Action Taken:** {message['action']}")
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Process the query with ICAR
            with st.spinner("ICAR analyzing your query..."):
                result = chatbot.process_query(prompt)
            
            # Add bot response to chat
            bot_message = {
                "role": "assistant",
                "content": result['response'],
                "reasoning": result['reasoning'],
                "action": result['action_taken']
            }
            st.session_state.messages.append(bot_message)
            
            # Rerun to update the chat display
            st.rerun()
    
    with col2:
        st.header("🧠 ICAR Example Queries")
        
        example_queries = [
            "Hello! How are you?",
            "What are the company work hours?",
            "Tell me about ACME Widget Pro features",
            "What is your return policy?",
            "What's the weather in London?",
            "How do I contact IT support?",
            "What benefits do employees get?",
            "What's the weather in Tokyo?"
        ]
        
        st.markdown("Try these ICAR-powered example queries:")
        for query in example_queries:
            if st.button(f"💬 {query}", key=f"example_{query[:20]}"):
                st.session_state.messages.append({"role": "user", "content": query})
                
                with st.spinner("ICAR processing..."):
                    result = chatbot.process_query(query)
                
                bot_message = {
                    "role": "assistant",
                    "content": result['response'],
                    "reasoning": result['reasoning'],
                    "action": result['action_taken']
                }
                st.session_state.messages.append(bot_message)
                st.rerun()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.header("🧠 How ICAR Works")
        st.markdown("""
        **ICAR Intelligence by Barış Genç:**
        
        1. **Concept Extraction**: ICAR analyzes your query to identify key concepts
        2. **Intelligent Matching**: It chooses the optimal strategy:
           - ICAR Direct Response for conversational queries
           - ICAR Concept-Based Retrieval for specific information
           - ICAR Semantic Search for complex queries
           - ICAR Weather API for weather information
        3. **Smart Execution**: Executes the chosen action with concept awareness
        4. **Transparent Analysis**: Shows detailed ICAR reasoning process
        
        **ICAR = Intelligent Concept-Aware RAG**
        """)

if __name__ == "__main__":
    main()