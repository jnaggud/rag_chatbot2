#!/usr/bin/env python
# main.py

import os
import sys
import subprocess
import argparse


# Ensure tokenizers warning suppressed
os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")

from utils.logging import setup_logging
logger = setup_logging()

from config.settings import (
    PERSIST_DIRECTORY,
    DESCRIPTIONS_FILE,
    DEFAULT_EMBEDDING_MODEL,
    AVAILABLE_LLM_MODELS,
    DEFAULT_LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COARSE_TOP_K,
    RERANK_TOP_K
)

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
    persist_directory: str = PERSIST_DIRECTORY,
    descriptions_file: str = DESCRIPTIONS_FILE,
    model_name: str = DEFAULT_LLM_MODEL
):
    # Install everything
    check_and_install_dependencies()
    check_and_install_ollama()
    pull_ollama_model(model_name)

    # Embeddings
    embedding_fn = get_embedding_function(DEFAULT_EMBEDDING_MODEL)

    # Chunking
    chunker = SemanticChunkingProcessor(
        embedding_fn,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # Vector DB
    vdb = VectorDatabaseManager(
        embedding_fn,
        persist_directory=persist_directory,
        descriptions_file=descriptions_file
    )

    # Index creation
    vdb.create_index("domain1", domain1_description)
    vdb.create_index("domain2", domain2_description)

    # Load + incremental chunk & index
    for name, data_dir in [("domain1", domain1_data_dir), ("domain2", domain2_data_dir)]:
        docs = load_documents(data_dir)
        if docs:
            chunks = chunker.chunk_documents(docs)
            vdb.add_documents_to_index(name, chunks)

    # Routing, reranking, retrieving
    router = QueryRouter(vdb, embedding_fn)
    reranker = SemanticReranker(embedding_fn, top_k=RERANK_TOP_K)
    retriever = RAGRetriever(
        vdb, router, reranker,
        coarse_top_k=COARSE_TOP_K,
        rerank_top_k=RERANK_TOP_K
    )

    # Chain & chatbot
    chain = RAGChain(retriever, model_name)
    bot = RAGChatbot(chain)

    logger.info("RAG Chatbot setup complete.")
    return bot


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
    #parser.add_argument("--model", type=str, default="llama3", help="Ollama model to use")
    from config.settings import DEFAULT_LLM_MODEL
    parser.add_argument(
    "--model",
    type=str,
    default=DEFAULT_LLM_MODEL,
    help="Ollama model to use"
    )
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
