# ICAR Chatbot Demo - Intelligent Concept-Aware RAG

An advanced intelligent chatbot implementing the **ICAR (Intelligent Concept-Aware RAG)** methodology. This system goes beyond traditional RAG by using concept extraction and intelligent matching to provide more accurate and contextually relevant responses.

**Author**: Barış Genç  
**Methodology**: ICAR (Intelligent Concept-Aware RAG)

## 🧠 ICAR Methodology

ICAR represents a significant advancement over traditional RAG systems:

- **Concept Extraction**: Automatically identifies key concepts from both documents and user queries
- **Intelligent Matching**: Uses multi-level concept matching (exact, semantic, categorical)
- **Smart Retrieval**: Prioritizes documents based on concept relevance rather than just keyword similarity
- **Enhanced Decision Making**: Makes intelligent choices about which retrieval strategy to use
- **Transparent Reasoning**: Provides detailed explanations of the decision-making process

## ✨ Key Features

- **🧠 Concept-Aware Intelligence**: ICAR methodology for superior understanding
- **🎯 Smart Action Selection**: Four specialized action types with intelligent routing
- **📊 Concept Indexing**: Advanced concept extraction and relationship mapping
- **🔍 Multi-Level Retrieval**: Concept-based, semantic, and direct response strategies
- **🌤️ Weather Integration**: Real-time weather data via OpenWeatherMap API
- **💬 Natural Conversations**: Enhanced greeting and conversation handling
- **📈 Performance Analytics**: Detailed statistics on concept extraction and matching
- **🖥️ Multiple Interfaces**: Both CLI and modern Streamlit web interface
- **🐳 Containerized**: Complete Docker support for easy deployment

## 🏗️ ICAR Architecture

The ICAR system uses four intelligent action types:

1. **ICAR Direct Response**: For greetings and simple conversational queries
2. **ICAR Concept-Based Retrieval**: Primary method using extracted concepts for precise matching
3. **ICAR Semantic Search**: Fallback method using traditional vector similarity
4. **ICAR Weather API**: Enhanced weather detection with concept awareness

### Concept Extraction Pipeline:
```
Document → Concept Extraction → Concept Index → Query Analysis → Concept Matching → Smart Retrieval → Enhanced Response
```

## Prerequisites

- Python 3.11+
- OpenAI API key
- OpenWeatherMap API key (optional, for weather functionality)
- Docker (optional, for containerized deployment)

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd ChatbotDemo
```

### 2. Environment Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your keys:
```
OPENAI_API_KEY=your_openai_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
```

### 3. Run with Docker (Recommended)

```bash
# Build and run with docker-compose
docker-compose up --build

# For CLI interface
docker-compose run chatbot python main.py

# For Streamlit interface
docker-compose up
# Then visit http://localhost:8501
```

### 4. Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Create sample documents
python create_sample_pdfs.py

# Run CLI interface
python main.py

# Or run Streamlit interface
streamlit run streamlit_app.py
```

## 🚀 ICAR Usage Examples

### CLI Interface with ICAR Intelligence

```
You: Hello!
ICAR Bot: Hello! I'm your ICAR-powered chatbot using Intelligent Concept-Aware RAG methodology. How can I help you today?

💭 ICAR Analysis: Simple greeting or basic query detected. No document retrieval needed.
🔧 Action Taken: ICAR Direct Response

You: What are the company work hours?
ICAR Bot: According to the company handbook, the work hours are Monday to Friday, 9:00 AM to 5:00 PM...

💭 ICAR Analysis: Detected high-confidence domain concepts: ['business']. Using intelligent concept-based retrieval.
🔧 Action Taken: ICAR Concept-Based Retrieval

You: What's the weather in London?
ICAR Bot: Weather in London, GB:
- Temperature: 15°C (feels like 13°C)
- Condition: Light Rain
- Humidity: 78%
...

💭 ICAR Analysis: Detected weather intent for 'London' with high concept confidence. Using weather API for real-time data.
🔧 Action Taken: ICAR Weather API Call

You: Tell me about product specifications
ICAR Bot: Based on concept analysis, here are the product specifications from our catalog...

💭 ICAR Analysis: Detected high-confidence domain concepts: ['product']. Using intelligent concept-based retrieval.
🔧 Action Taken: ICAR Concept-Based Retrieval
```

### ICAR Streamlit Interface

The enhanced web interface provides:
- Real-time chat with ICAR-powered responses
- **Concept Analysis Transparency**: See which concepts were extracted and matched
- **Decision Process Visualization**: Understand ICAR's reasoning
- **Performance Metrics**: View concept extraction statistics
- Example queries showcasing ICAR capabilities
- System status with concept index information

## Project Structure

```
ChatbotDemo/
├── src/
│   ├── agent_core.py      # Main agentic logic
│   ├── vector_store.py    # ChromaDB integration
│   ├── document_processor.py  # PDF processing
│   └── external_apis.py   # Weather API integration
├── data/                  # PDF documents for knowledge base
├── tests/                 # Test files
├── main.py               # CLI interface
├── streamlit_app.py      # Web interface
├── create_sample_pdfs.py # Sample data generator
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker compose setup
└── .env.example         # Environment template
```

## How It Works

### Decision Making Process

1. **Query Analysis**: The agent analyzes the user input for patterns
2. **Action Selection**: Based on keywords and context, it chooses:
   - **Direct Response**: Greetings, simple questions
   - **Vector Search**: Document-specific queries
   - **Weather API**: Weather-related questions
3. **Execution**: Performs the selected action
4. **Response**: Provides answer with reasoning transparency

### Knowledge Base

- PDF documents are processed and chunked
- Text embeddings created using OpenAI
- Stored in ChromaDB vector database
- Similarity search for relevant context
- RAG response generation

### External APIs

- OpenWeatherMap for real-time weather data
- Automatic city extraction from queries
- Error handling and fallbacks
- Response formatting

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Required for LLM and embeddings
- `OPENWEATHER_API_KEY`: Optional for weather functionality

### Customization

- Modify decision patterns in `src/agent_core.py`
- Add new document types in `src/document_processor.py`
- Extend external APIs in `src/external_apis.py`
- Adjust chunk sizes and embedding parameters

## 🛠️ ICAR Development

### Running ICAR Tests

```bash
# Run comprehensive ICAR test suite
python test_icar.py

# Run basic functionality tests
python test_icar.py --basic

# Run legacy tests (optional)
python test_basic.py
python test_structure.py
python test_minimal.py
```

### Adding New Features

1. Extend the `ActionType` enum in `agent_core.py`
2. Add decision logic in `decide_action()` method
3. Implement execution method
4. Update the main processing loop

### Custom Documents

Place PDF files in the `data/` directory. They will be automatically processed and added to the knowledge base.

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure `.env` file exists with valid keys
2. **No Documents**: Run `python create_sample_pdfs.py` to create samples
3. **Docker Issues**: Check if ports 8501 is available
4. **Import Errors**: Verify all requirements are installed

### Logs

Check `chatbot.log` for detailed logging information.

## 📊 ICAR Performance

The ICAR system provides enhanced performance metrics:
- **Concept Extraction Rate**: Number of concepts extracted per document
- **Matching Accuracy**: Precision of concept-based retrieval
- **Response Quality**: Improved relevance through intelligent matching
- **Processing Speed**: Optimized for real-time concept analysis

## 📄 License

This ICAR implementation is for demonstration and research purposes. 
**Author**: Barış Genç  
**Methodology**: ICAR (Intelligent Concept-Aware RAG)

Refer to individual package licenses for dependencies.

## 🤝 Support

**ICAR System**: Developed by Barış Genç  
For technical questions about the ICAR methodology or implementation, please check the logs and ensure all ICAR dependencies are properly installed.

---

*Powered by ICAR (Intelligent Concept-Aware RAG) - A revolutionary approach to document retrieval and question answering by Barış Genç.*