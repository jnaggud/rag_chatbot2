#!/usr/bin/env python
"""
RAG Chatbot with Semantic Chunking, Reranking, and Query Routing
Main entry point for the application.
"""

import argparse
import sys
import subprocess
import os

# Set up logging first
from utils.logging import setup_logging
logger = setup_logging()

# Import all required components
from utils.dependencies import check_and_install_dependencies
from models.ollama import check_and_install_ollama, pull_ollama_model
from rag.document_loaders import load_documents
from rag.chunking import SemanticChunkingProcessor
from rag.embeddings import get_embedding_function
from rag.vector_store import VectorDatabaseManager
from rag.query_router import QueryRouter
from rag.reranker import SemanticReranker
from rag.retriever import RAGRetriever
from rag.chain import RAGChain
from interfaces.chatbot import RAGChatbot

def setup_rag_chatbot(
    domain1_data_dir: str,
    domain2_data_dir: str,
    domain1_description: str,
    domain2_description: str,
    persist_directory: str = "./chroma_db",
    model_name: str = "llama3"
):
    """
    Set up the complete RAG chatbot system.
    
    Args:
        domain1_data_dir: Directory containing documents for the first domain
        domain2_data_dir: Directory containing documents for the second domain
        domain1_description: Description of the first domain for query routing
        domain2_description: Description of the second domain for query routing
        persist_directory: Directory to persist the vector database
        model_name: Name of the Ollama model to use
    
    Returns:
        The initialized chatbot instance
    """
    # Check and install dependencies
    check_and_install_dependencies()
    
    # Check and install Ollama
    check_and_install_ollama()
    
    # Pull the LLM model
    pull_ollama_model(model_name)
    
    # Create embedding function
    embedding_function = get_embedding_function()
    
    # Initialize semantic chunking processor
    chunking_processor = SemanticChunkingProcessor(embedding_function)
    
    # Initialize vector database manager
    vector_db_manager = VectorDatabaseManager(embedding_function, persist_directory)
    
    # Create indexes for the two domains
    vector_db_manager.create_index("domain1", domain1_description)
    vector_db_manager.create_index("domain2", domain2_description)
    
    # Load and process documents for domain 1
    domain1_docs = load_documents(domain1_data_dir)
    if domain1_docs:
        chunked_domain1_docs = chunking_processor.chunk_documents(domain1_docs)
        vector_db_manager.add_documents_to_index("domain1", chunked_domain1_docs)
    
    # Load and process documents for domain 2
    domain2_docs = load_documents(domain2_data_dir)
    if domain2_docs:
        chunked_domain2_docs = chunking_processor.chunk_documents(domain2_docs)
        vector_db_manager.add_documents_to_index("domain2", chunked_domain2_docs)
    
    # Initialize query router
    query_router = QueryRouter(vector_db_manager, embedding_function)
    
    # Initialize semantic reranker
    semantic_reranker = SemanticReranker(embedding_function)
    
    # Initialize RAG retriever
    rag_retriever = RAGRetriever(vector_db_manager, query_router, semantic_reranker)
    
    # Initialize RAG chain
    rag_chain = RAGChain(rag_retriever, model_name)
    
    # Initialize chatbot
    chatbot = RAGChatbot(rag_chain)
    
    logger.info("RAG Chatbot setup complete. Ready to chat!")
    
    return chatbot

def run_streamlit_directly():
    """Run Streamlit directly without importing the module"""
    # Set critical environment variables
    os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = "false"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    
    streamlit_script = os.path.join("interfaces", "streamlit_app.py")
    
    print("\n" + "="*80)
    print("Starting Streamlit server...")
    print("If the browser doesn't open automatically, manually go to: http://localhost:8501")
    print("="*80 + "\n")
    
    # Simple command with minimal options
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        streamlit_script,
        "--server.port=8501"
    ]
    
    subprocess.run(cmd)
    return

def main():
    """Main function to parse arguments and run the appropriate interface."""
    parser = argparse.ArgumentParser(description="RAG Chatbot with Semantic Chunking, Reranking, and Query Routing")
    parser.add_argument("--domain1", type=str, required=False, default="./data/domain1", help="Directory containing documents for domain 1")
    parser.add_argument("--domain2", type=str, required=False, default="./data/domain2", help="Directory containing documents for domain 2")
    parser.add_argument("--desc1", type=str, required=False, default="First domain description", help="Description of domain 1")
    parser.add_argument("--desc2", type=str, required=False, default="Second domain description", help="Description of domain 2")
    parser.add_argument("--persist", type=str, default="./chroma_db", help="Directory to persist vector database")
    parser.add_argument("--model", type=str, default="llama3", help="Ollama model to use")
    parser.add_argument("--web", action="store_true", help="Launch web interface instead of CLI")
    parser.add_argument("--streamlit", action="store_true", help="Launch Streamlit interface instead of CLI or web")
    
    args = parser.parse_args()
    
    # Launch Streamlit if requested - Use direct method instead of importing
    if args.streamlit:
        run_streamlit_directly()
        return
    
    # Setup chatbot for CLI or web interface
    chatbot = setup_rag_chatbot(
        domain1_data_dir=args.domain1,
        domain2_data_dir=args.domain2,
        domain1_description=args.desc1,
        domain2_description=args.desc2,
        persist_directory=args.persist,
        model_name=args.model
    )
    
    # Run interface
    if args.web:
        from interfaces.web import create_web_interface
        create_web_interface(chatbot)
    else:
        from interfaces.cli import run_cli
        run_cli(chatbot)

if __name__ == "__main__":
    main()
