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
Bot: Hello! I'm your ICAR-powered chatbot using Intelligent Concept-Aware RAG methodology. How can I help you today?

💭 ICAR Analysis: Simple greeting or basic query detected. No document retrieval needed.
🔧 Action Taken: ICAR Direct Response

You: What are the company work hours?
Bot: According to the company handbook, the work hours are Monday to Friday, 9:00 AM to 5:00 PM...

💭 ICAR Analysis: Detected high-confidence domain concepts: ['employee_policies']. Using intelligent concept-based retrieval.
🔧 Action Taken: ICAR Concept-Based Retrieval

You: What's the weather in London?
Bot: Weather in London, GB:
- Temperature: 15°C (feels like 13°C)
- Condition: Light Rain
- Humidity: 78%
...

💭 ICAR Analysis: Detected weather intent for 'London' with high concept confidence. Using weather API for real-time data.
🔧 Action Taken: ICAR Weather API Call

You: Tell me about product specifications
Bot: Based on concept analysis, here are the product specifications from our catalog...

💭 ICAR Analysis: Detected high-confidence domain concepts: ['products']. Using intelligent concept-based retrieval.
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

## 📁 ICAR Project Structure

```
ChatbotDemo/
├── src/
│   ├── agent_core.py       # ICAR methodology implementation
│   ├── concept_extractor.py # Core concept extraction & matching
│   ├── vector_store.py     # Enhanced ChromaDB integration
│   ├── document_processor.py # PDF processing with concepts
│   └── external_apis.py    # Weather API integration
├── data/                   # PDF documents for knowledge base
├── tests/                  # ICAR system tests
├── main.py                # ICAR-powered CLI interface
├── streamlit_app.py       # ICAR-enhanced web interface
├── create_sample_pdfs.py  # Sample data generator
├── concept_index.json     # Generated concept index
├── requirements.txt       # Enhanced Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker compose setup
└── .env.example         # Environment template
```

## 🔬 How ICAR Works

### ICAR Decision Making Process

1. **Query Concept Extraction**: Advanced analysis to identify user intent and key concepts
2. **Intelligent Action Selection**: ICAR chooses the optimal strategy:
   - **ICAR Direct Response**: For conversational queries
   - **ICAR Concept-Based Retrieval**: Primary method using concept matching
   - **ICAR Semantic Search**: Enhanced fallback with traditional similarity
   - **ICAR Weather API**: Concept-aware weather detection
3. **Smart Execution**: Performs the selected action with concept awareness
4. **Enhanced Response**: Provides answer with detailed ICAR reasoning

### ICAR Knowledge Processing

- **Concept Extraction**: Documents processed to identify key concepts
- **Concept Indexing**: Builds relationships between concepts and documents
- **Multi-Level Matching**: Exact, semantic, and categorical concept matching
- **Smart Retrieval**: Prioritizes documents based on concept relevance
- **Continuous Learning**: Concept index improves over time

### ICAR Technology Stack

- **Concept Processing**: Advanced NLP with spaCy and custom algorithms
- **Concept Storage**: NetworkX for relationship mapping
- **Enhanced Embeddings**: OpenAI embeddings with concept weighting
- **Vector Database**: ChromaDB with concept-enhanced metadata
- **LLM Integration**: GPT-3.5-turbo with ICAR-optimized prompts

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Required for LLM and embeddings
- `OPENWEATHER_API_KEY`: Optional for weather functionality

### ICAR Customization

- **Concept Categories**: Modify concept categories in `src/concept_extractor.py`
- **Decision Logic**: Enhance ICAR decision patterns in `src/agent_core.py`
- **Concept Extraction**: Adjust concept extraction algorithms for your domain
- **Matching Strategies**: Fine-tune concept matching thresholds and weights
- **Document Processing**: Add new document types with concept extraction support

## 🛠️ ICAR Development

### Running ICAR Tests

```bash
# Run all tests including ICAR functionality
python test_basic.py
python test_structure.py
python test_minimal.py

# Test concept extraction specifically
python -c "from src.concept_extractor import ConceptExtractor; ce = ConceptExtractor(); print('ICAR tests passed')"
```

### Extending ICAR

1. **Add New Action Types**: Extend the `ActionType` enum in `agent_core.py`
2. **Enhance Concepts**: Add new concept categories in `concept_extractor.py`
3. **Improve Matching**: Adjust concept matching algorithms for better precision
4. **Add Intelligence**: Implement new decision logic in `decide_action()` method

### ICAR Document Processing

Place PDF files in the `data/` directory. ICAR will automatically:
1. Extract text content
2. Identify key concepts
3. Build concept relationships
4. Create searchable index
5. Enable intelligent retrieval

## 🔧 Troubleshooting

### ICAR-Specific Issues

1. **Concept Extraction Fails**: Check OpenAI API limits and connection
2. **No Concept Index**: Ensure documents are loaded and processed
3. **Poor Concept Matching**: Adjust concept weights and thresholds
4. **Missing Dependencies**: Install spaCy and NetworkX for concept processing

### General Issues

1. **Missing API Keys**: Ensure `.env` file exists with valid keys
2. **No Documents**: Run `python create_sample_pdfs.py` to create samples
3. **Docker Issues**: Check if port 8501 is available
4. **Import Errors**: Verify all requirements including ICAR dependencies

### Logs and Debugging

- Check `chatbot.log` for ICAR processing information
- Look for concept extraction statistics in logs
- Monitor concept index generation progress
- Verify concept matching scores in debug output

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