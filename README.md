cat > README.md << 'EOF'
# RAG Chatbot with Semantic Chunking, Reranking, and Query Routing

A retrieval-augmented generation (RAG) chatbot that uses advanced techniques including semantic chunking, multi-domain knowledge with query routing, and semantic reranking.

## Features

- **Semantic Chunking**: Divides documents based on meaning rather than arbitrary character counts
- **Multiple Knowledge Domains**: Organizes knowledge into separate vector indexes
- **Query Routing**: Automatically directs questions to the relevant knowledge domain
- **Semantic Reranking**: Improves search results by re-ranking based on semantic similarity

## Prerequisites

- Python 3.9+ (3.10 recommended)
- [Ollama](https://ollama.com/) for running LLMs locally

## Installation

1. Clone the repository: https://github.com/jnaggud/rag_chatbot.git

2. Create and activate a virtual environment: python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies: pip install -r requirements.txt

4. Start Ollama in a separate terminal: ollama serve

5. Pull the required model: ollama pull llama3

## Usage

1. Add your PDF documents to the data directories:
- `data/domain1/` - First knowledge domain
- `data/domain2/` - Second knowledge domain

2. Run the application with Streamlit interface: python main.py --streamlit

## Project Structure

- `config/`: Configuration settings
- `rag/`: Core RAG components
- `models/`: Model management
- `utils/`: Utility functions
- `interfaces/`: User interfaces
- `data/`: Document directories