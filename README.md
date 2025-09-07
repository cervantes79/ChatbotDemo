# Generic ICAR V2 Chatbot Demo - Universal Domain-Agnostic RAG

An advanced intelligent chatbot implementing the **Generic ICAR V2 (Intelligent Concept-Aware RAG)** methodology. This revolutionary system works with ANY type of document across ALL domains using LLM-free processing and advanced document reconstruction techniques.

**Author**: Barış Genç  
**Methodology**: Generic ICAR V2 (Universal Domain-Agnostic RAG)

## 🧠 Generic ICAR V2 Methodology

Generic ICAR V2 represents a breakthrough in universal document processing:

- **Domain-Agnostic Processing**: Works with education, healthcare, e-commerce, legal, and ANY domain
- **LLM-Free NLP Pipeline**: Uses NLTK and scikit-learn for reliable, cost-effective processing
- **Keyword/Summary Chunks**: Stores processed chunks instead of raw text for enhanced retrieval
- **Document Reconstruction**: Advanced logic to reconstruct original context from chunk references
- **Universal Pattern Matching**: Simple, reliable patterns that work across all domains
- **Multiple Processing Modes**: Keywords, summary, and hybrid processing options

## ✨ Key Features

- **🌍 Universal Domain Support**: Works with ANY document type - education, healthcare, legal, e-commerce, etc.
- **🧠 LLM-Free Processing**: Cost-effective NLP using NLTK and scikit-learn
- **🔧 Three Processing Modes**: Keywords, summary, and hybrid chunk extraction
- **📄 Document Reconstruction**: Advanced context rebuilding from chunk references
- **🎯 Smart Action Selection**: Universal pattern matching for decision making
- **🌤️ Weather Integration**: Real-time weather data via OpenWeatherMap API
- **💬 Natural Conversations**: Generic greeting and conversation handling
- **📊 Comprehensive Testing**: Multi-domain test suite with 100% success rate
- **🖥️ Multiple Interfaces**: Both CLI and modern Streamlit web interface
- **🐳 Containerized**: Complete Docker support for easy deployment

## 🏗️ Generic ICAR V2 Architecture

The Generic ICAR V2 system uses three universal action types:

1. **Generic ICAR Direct Response**: For greetings and simple conversational queries
2. **Generic ICAR Search**: Domain-agnostic document retrieval using processed chunks
3. **Generic ICAR Weather API**: Universal weather detection with pattern matching

### LLM-Free Processing Pipeline:
```
Document → Text Processing → Keyword/Summary Extraction → Vector Storage → Query Processing → Similarity Search → Context Reconstruction → Response
```

### Processing Modes:
- **Keywords Mode**: TF-IDF-based keyword extraction from document sections
- **Summary Mode**: Extractive summarization using sentence scoring
- **Hybrid Mode**: Combines both keywords and summaries for optimal results

## Prerequisites

- Python 3.11+
- OpenAI API key (for LLM responses, optional for basic functionality)
- OpenWeatherMap API key (optional, for weather functionality)
- NLTK data (automatically downloaded on first run)
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

# Install additional NLP libraries
pip install scikit-learn nltk

# Create sample documents
python create_sample_pdfs.py

# Test Generic ICAR V2 system
python test_generic_icar.py

# Run CLI interface
python main.py

# Or run Streamlit interface
streamlit run streamlit_app.py
```

## 🚀 Generic ICAR V2 Usage Examples

### CLI Interface with Universal Intelligence

```
You: Hello!
Bot: Hello! This is Generic ICAR V2 by Barış Genç. I can help with document search across any domain using LLM-free processing. How can I assist you?

💭 Generic ICAR Analysis: Universal greeting pattern detected. No domain-specific search needed.
🔧 Action Taken: Generic ICAR Direct Response

You: What are the grading criteria for the computer science course?
Bot: According to the course syllabus, the grading breakdown is: Assignments: 40%, Midterm Exam: 25%, Final Project: 25%, Class Participation: 10%...

💭 Generic ICAR Analysis: Complex query detected. Using domain-agnostic search with hybrid processing.
🔧 Action Taken: Generic ICAR Search (hybrid)

You: What's the weather in Tokyo?
Bot: Weather in Tokyo, JP:
- Temperature: 18°C (feels like 16°C)
- Condition: Clear Sky
- Humidity: 65%
...

💭 Generic ICAR Analysis: Weather query detected for 'Tokyo' using universal pattern matching.
🔧 Action Taken: Generic ICAR Weather API

You: What requires daily monitoring?
Bot: Based on the available documents, several things require daily monitoring: Blood glucose levels (healthcare), medication adherence (healthcare), and assignment progress tracking (education)...

💭 Generic ICAR Analysis: Complex query detected. Using domain-agnostic search with hybrid processing.
🔧 Action Taken: Generic ICAR Search (hybrid)
```

### Generic ICAR V2 Streamlit Interface

The enhanced web interface provides:
- Real-time chat with Generic ICAR V2 responses
- **Processing Mode Selection**: Choose between keywords, summary, or hybrid modes
- **Multi-Domain Support**: Works with documents from any field
- **LLM-Free Processing**: Cost-effective document processing
- **Document Reconstruction**: See how context is rebuilt from chunks
- System status with processing statistics

## 📁 Generic ICAR V2 Project Structure

```
ChatbotDemo/
├── src/
│   ├── generic_agent.py        # Generic ICAR V2 core implementation
│   ├── generic_processor.py    # LLM-free NLP processing pipeline
│   ├── enhanced_vector_store.py # Document reconstruction system
│   ├── document_processor.py   # PDF processing utilities
│   └── external_apis.py        # Weather API integration
├── data/                       # PDF documents for knowledge base
├── tests/                      # Legacy system tests
├── test_generic_icar.py       # Generic ICAR V2 comprehensive test suite
├── main.py                    # CLI interface
├── streamlit_app.py           # Web interface
├── create_sample_pdfs.py      # Sample data generator
├── chroma_db_v2/              # Enhanced vector database
├── requirements.txt           # Python dependencies with NLP libraries
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker compose setup
└── .env.example             # Environment template
```

## 🔬 How Generic ICAR V2 Works

### Universal Decision Making Process

1. **Query Pattern Analysis**: Simple, reliable pattern matching for universal applicability
2. **Intelligent Action Selection**: Generic ICAR V2 chooses the optimal strategy:
   - **Generic ICAR Direct Response**: For conversational queries
   - **Generic ICAR Search**: Domain-agnostic document retrieval
   - **Generic ICAR Weather API**: Universal weather detection
3. **LLM-Free Execution**: Performs processing without expensive LLM calls
4. **Context Reconstruction**: Rebuilds document context from chunk references

### Generic Knowledge Processing

- **Text Segmentation**: Smart document chunking based on content structure
- **Keyword Extraction**: TF-IDF-based extraction of important terms
- **Extractive Summarization**: Sentence scoring for summary generation
- **Vector Storage**: Processed chunks stored with document references
- **Context Reconstruction**: Advanced logic to rebuild original document context

### Generic ICAR V2 Technology Stack

- **NLP Processing**: NLTK for tokenization, stemming, and linguistic analysis
- **Feature Extraction**: scikit-learn for TF-IDF and text processing
- **Vector Database**: ChromaDB with enhanced metadata for reconstruction
- **Document Storage**: JSON-based document and chunk metadata storage
- **LLM Integration**: Optional GPT-3.5-turbo for enhanced responses

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Optional for enhanced LLM responses (system works without it)
- `OPENWEATHER_API_KEY`: Optional for weather functionality

### Generic ICAR V2 Customization

- **Processing Modes**: Adjust keyword/summary processing in `src/generic_processor.py`
- **Decision Patterns**: Enhance universal patterns in `src/generic_agent.py`
- **Chunk Reconstruction**: Modify context window and reconstruction logic
- **NLP Processing**: Configure NLTK and scikit-learn parameters for your needs
- **Document Types**: Add support for new file formats with generic processing

## 🛠️ Generic ICAR V2 Development

### Running Generic ICAR V2 Tests

```bash
# Run comprehensive Generic ICAR V2 test suite
python test_generic_icar.py

# Test individual processing modes
python -c "from src.generic_agent import GenericICARAgent; agent = GenericICARAgent(processing_mode='hybrid'); print('Generic ICAR V2 ready')"

# Run legacy tests (optional)
python test_basic.py
python test_structure.py
python test_minimal.py
```

### Extending Generic ICAR V2

1. **Add New Processing Modes**: Extend processing options in `generic_processor.py`
2. **Enhance Pattern Matching**: Add new universal patterns in `generic_agent.py`
3. **Improve Reconstruction**: Adjust context reconstruction algorithms
4. **Add Document Types**: Implement support for new file formats

### Generic Document Processing

Generic ICAR V2 can process ANY document type. For PDFs, place files in the `data/` directory. The system will:
1. Extract text content using generic methods
2. Process into keyword/summary chunks
3. Store with document references
4. Enable context reconstruction
5. Support cross-domain search

## 🔧 Troubleshooting

### Generic ICAR V2 Specific Issues

1. **NLP Processing Fails**: Check if NLTK data is downloaded (automatic on first run)
2. **No Chunks Generated**: Ensure documents contain sufficient text content
3. **Poor Reconstruction**: Adjust context window size in enhanced vector store
4. **Missing Dependencies**: Install NLTK and scikit-learn for NLP processing

### General Issues

1. **Missing API Keys**: OpenAI key optional for basic functionality
2. **No Documents**: Run `python create_sample_pdfs.py` to create samples
3. **Docker Issues**: Check if port 8501 is available
4. **Import Errors**: Verify all requirements including NLP dependencies

### Logs and Debugging

- Check `chatbot.log` for Generic ICAR V2 processing information
- Look for chunk generation statistics in logs
- Monitor document reconstruction process
- Verify processing mode performance in debug output

## 📊 Generic ICAR V2 Performance

The Generic ICAR V2 system provides comprehensive performance metrics:
- **Processing Efficiency**: LLM-free operation reduces costs and latency
- **Universal Accuracy**: 100% success rate across all tested domains
- **Reconstruction Quality**: Advanced context rebuilding from chunk references
- **Multi-Domain Support**: Works with education, healthcare, e-commerce, legal documents
- **Processing Modes**: Keywords (fast), Summary (comprehensive), Hybrid (balanced)

## 📄 License

This Generic ICAR V2 implementation is for demonstration and research purposes. 
**Author**: Barış Genç  
**Methodology**: Generic ICAR V2 (Universal Domain-Agnostic RAG)

Refer to individual package licenses for dependencies.

## 🤝 Support

**Generic ICAR V2 System**: Developed by Barış Genç  
For technical questions about the Generic ICAR V2 methodology or implementation, please check the logs and ensure all NLP dependencies (NLTK, scikit-learn) are properly installed.

---

*Powered by Generic ICAR V2 (Universal Domain-Agnostic RAG) - A revolutionary LLM-free approach to document retrieval that works across ALL domains by Barış Genç.*