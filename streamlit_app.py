import streamlit as st
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from src.agent_core import AgenticChatbot

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Chatbot Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot (cached for performance)"""
    load_dotenv()
    
    # Check required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found. Please set it in your .env file.")
        return None
    
    chatbot = AgenticChatbot()
    
    if not chatbot.initialize():
        st.error("Failed to initialize chatbot")
        return None
    
    # Load documents if available
    data_dir = "data"
    if os.path.exists(data_dir):
        with st.spinner("Loading documents into knowledge base..."):
            success = chatbot.load_documents(data_dir)
            if success:
                stats = chatbot.get_stats()
                st.success(f"Knowledge base loaded with {stats['documents_count']} document chunks")
            else:
                st.warning("Failed to load documents, but chatbot will still work for basic queries")
    
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
    st.title("🤖 Agentic RAG Chatbot Demo")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("System Information")
        
        # Initialize chatbot
        chatbot = initialize_chatbot()
        
        if chatbot:
            stats = chatbot.get_stats()
            st.success("✅ Chatbot Initialized")
            st.info(f"📄 Documents: {stats.get('documents_count', 0)}")
            st.info(f"🗄️ Vector Store: {'✅' if stats.get('vector_store_ready', False) else '❌'}")
            st.info(f"🌤️ Weather API: {'✅' if stats.get('weather_api_ready', False) else '❌'}")
        else:
            st.error("❌ Chatbot Not Initialized")
            st.stop()
        
        st.markdown("---")
        st.header("Capabilities")
        st.markdown("""
        This chatbot can:
        - 📚 **Knowledge Base Search**: Answer questions using loaded documents
        - 🌤️ **Weather Information**: Get current weather for any city
        - 💬 **General Conversation**: Respond to greetings and basic questions
        
        The bot explains its decision-making process for each response.
        """)
        
        # Sample data creation
        st.markdown("---")
        if st.button("🔄 Create Sample Documents"):
            create_sample_data()
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chat Interface")
        
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
                            with st.expander("🧠 Decision Process"):
                                st.write(f"**Reasoning:** {message['reasoning']}")
                                st.write(f"**Action Taken:** {message['action']}")
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Process the query
            with st.spinner("Thinking..."):
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
        st.header("Example Queries")
        
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
        
        st.markdown("Try these example queries:")
        for query in example_queries:
            if st.button(f"💬 {query}", key=f"example_{query[:20]}"):
                st.session_state.messages.append({"role": "user", "content": query})
                
                with st.spinner("Processing..."):
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
        st.header("How It Works")
        st.markdown("""
        **Agentic Decision Making:**
        
        1. **Input Analysis**: The bot analyzes your query
        2. **Action Selection**: It decides which tool to use:
           - Direct response for greetings
           - Knowledge base for specific info
           - Weather API for weather queries
        3. **Response Generation**: Executes the chosen action
        4. **Transparency**: Shows its reasoning process
        """)

if __name__ == "__main__":
    main()