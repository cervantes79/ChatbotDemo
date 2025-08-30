# Agentic RAG Chatbot Demo

An intelligent chatbot that demonstrates agentic behavior by automatically deciding which tools to use based on user queries. It combines Retrieval-Augmented Generation (RAG) with external API calls and direct responses.

## Features

- **Agentic Decision Making**: Automatically chooses the best approach for each query
- **Knowledge Base Search**: RAG implementation using ChromaDB and OpenAI embeddings
- **Weather Information**: Real-time weather data via OpenWeatherMap API
- **Direct Conversations**: Handles greetings and general questions
- **Transparent Reasoning**: Shows the decision-making process for each response
- **Multiple Interfaces**: Both CLI and Streamlit web interface
- **Dockerized**: Easy deployment with Docker and docker-compose

## Architecture

The chatbot uses three main action types:

1. **Direct Response**: For greetings and simple questions
2. **Vector Search**: For knowledge base queries using RAG
3. **Weather API**: For weather-related questions

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

## Usage Examples

### CLI Interface

```
You: Hello!
Bot: Hello! I'm your Agentic RAG Chatbot. How can I help you today?

💭 Decision Process: This appears to be a simple greeting...
🔧 Action Taken: Direct Response

You: What are the company work hours?
Bot: According to the company handbook, the work hours are Monday to Friday, 9:00 AM to 5:00 PM...

💭 Decision Process: The query contains keywords that suggest...
🔧 Action Taken: Knowledge Base Search

You: What's the weather in London?
Bot: Weather in London, GB:
- Temperature: 15°C (feels like 13°C)
- Condition: Light Rain
- Humidity: 78%
...

💭 Decision Process: User is asking about weather information for 'London'...
🔧 Action Taken: Weather API Call
```

### Streamlit Interface

The web interface provides:
- Real-time chat with message history
- Decision process transparency
- Example queries to try
- System status information
- Sample document generation

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

## Development

### Running Tests

```bash
python -m pytest tests/
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

## License

This project is for demonstration purposes. Refer to individual package licenses for dependencies.

## Support

For issues and questions, please check the logs and ensure all dependencies are properly installed.