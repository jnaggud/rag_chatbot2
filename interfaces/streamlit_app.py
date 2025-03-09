"""
Streamlit interface for the RAG chatbot.
"""

import streamlit as st
import sys
import os
import logging
import time
from typing import List, Dict
import traceback

# Set up page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Add the parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules (after adding parent to path)
from utils.logging import setup_logging
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

# Set up logging
logger = setup_logging()

# Configure Streamlit logging
streamlit_logger = logging.getLogger("streamlit")
streamlit_logger.setLevel(logging.WARNING)  # Reduce noise from Streamlit logs

def check_ollama_running():
    """Check if Ollama service is running and provide instructions if not."""
    import subprocess
    import platform
    
    try:
        # Different check commands based on OS
        if platform.system() == "Windows":
            cmd = "tasklist | findstr ollama"
        else:  # macOS and Linux
            cmd = "ps aux | grep ollama | grep -v grep"

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if not result.stdout.strip():
            st.error("""
            Ollama service does not appear to be running! 
            
            Please start Ollama by opening a new terminal and running:
            ```
            ollama serve
            ```
            
            Then refresh this page.
            """)
            return False
        return True
    except Exception as e:
        st.error(f"Error checking Ollama status: {e}")
        return False

def setup_rag_chatbot():
    """
    Set up the complete RAG chatbot system with values from Streamlit sidebar.
    """
    # Get values from session state
    domain1_data_dir = st.session_state.domain1_dir
    domain2_data_dir = st.session_state.domain2_dir
    domain1_description = st.session_state.domain1_desc
    domain2_description = st.session_state.domain2_desc
    persist_directory = st.session_state.persist_dir
    model_name = st.session_state.model_name
    
    # Check if Ollama is running
    if not check_ollama_running():
        return False
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Check dependencies
        status_text.text("Checking dependencies...")
        progress_bar.progress(5)
        check_and_install_dependencies()
        
        # Step 2: Check Ollama
        status_text.text("Checking Ollama installation...")
        progress_bar.progress(10)
        check_and_install_ollama()
        
        # Step 3: Pull model
        status_text.text(f"Pulling LLM model: {model_name}...")
        progress_bar.progress(20)
        pull_ollama_model(model_name)
        
        # Step 4: Create embedding function
        status_text.text("Setting up embeddings...")
        progress_bar.progress(30)
        embedding_function = get_embedding_function()
        
        # Step 5: Initialize semantic chunking processor
        status_text.text("Initializing semantic chunking...")
        progress_bar.progress(35)
        chunking_processor = SemanticChunkingProcessor(embedding_function)
        
        # Step 6: Initialize vector database manager
        status_text.text("Setting up vector database...")
        progress_bar.progress(40)
        vector_db_manager = VectorDatabaseManager(embedding_function, persist_directory)
        
        # Step 7: Create indexes
        status_text.text("Creating vector indexes...")
        progress_bar.progress(45)
        vector_db_manager.create_index("domain1", domain1_description)
        vector_db_manager.create_index("domain2", domain2_description)
        
        # Step 8: Load and process documents for domain 1
        status_text.text(f"Loading documents from {domain1_data_dir}...")
        progress_bar.progress(50)
        domain1_docs = load_documents(domain1_data_dir)
        if domain1_docs:
            status_text.text(f"Semantically chunking {len(domain1_docs)} documents for domain 1...")
            progress_bar.progress(60)
            chunked_domain1_docs = chunking_processor.chunk_documents(domain1_docs)
            status_text.text(f"Adding {len(chunked_domain1_docs)} chunks to domain 1 index...")
            progress_bar.progress(65)
            vector_db_manager.add_documents_to_index("domain1", chunked_domain1_docs)
        
        # Step 9: Load and process documents for domain 2
        status_text.text(f"Loading documents from {domain2_data_dir}...")
        progress_bar.progress(70)
        domain2_docs = load_documents(domain2_data_dir)
        if domain2_docs:
            status_text.text(f"Semantically chunking {len(domain2_docs)} documents for domain 2...")
            progress_bar.progress(80)
            chunked_domain2_docs = chunking_processor.chunk_documents(domain2_docs)
            status_text.text(f"Adding {len(chunked_domain2_docs)} chunks to domain 2 index...")
            progress_bar.progress(85)
            vector_db_manager.add_documents_to_index("domain2", chunked_domain2_docs)
        
        # Step 10: Initialize query router
        status_text.text("Setting up query routing...")
        progress_bar.progress(90)
        query_router = QueryRouter(vector_db_manager, embedding_function)
        
        # Step 11: Initialize semantic reranker
        status_text.text("Setting up semantic reranking...")
        progress_bar.progress(92)
        semantic_reranker = SemanticReranker(embedding_function)
        
        # Step 12: Initialize RAG retriever
        status_text.text("Building retrieval system...")
        progress_bar.progress(94)
        rag_retriever = RAGRetriever(vector_db_manager, query_router, semantic_reranker)
        
        # Step 13: Initialize RAG chain
        status_text.text("Building RAG chain with LLM...")
        progress_bar.progress(96)
        rag_chain = RAGChain(rag_retriever, model_name)
        
        # Step 14: Initialize chatbot
        status_text.text("Finalizing chatbot setup...")
        progress_bar.progress(98)
        chatbot = RAGChatbot(rag_chain)
        
        st.session_state.chatbot = chatbot
        st.session_state.setup_complete = True
        
        progress_bar.progress(100)
        status_text.text("RAG Chatbot setup complete!")
        time.sleep(1)  # Give user a moment to see the completion message
        status_text.empty()
        progress_bar.empty()
        
        st.success("RAG Chatbot setup complete! You can now chat with your documents.")
        return True
        
    except Exception as e:
        progress_bar.empty()
        error_msg = str(e)
        st.error(f"Error setting up RAG chatbot: {error_msg}")
        logger.error(f"Error setting up RAG chatbot: {error_msg}")
        logger.error(traceback.format_exc())
        return False

def display_debug_info():
    """Display debug information for troubleshooting."""
    with st.expander("Debug Information", expanded=False):
        st.write("### System Information")
        
        import platform
        import sys
        import subprocess
        
        st.text(f"Python Version: {sys.version}")
        st.text(f"Platform: {platform.platform()}")
        
        st.write("### Installed Packages")
        try:
            packages = subprocess.check_output([sys.executable, "-m", "pip", "list"]).decode()
            st.code(packages)
        except Exception as e:
            st.error(f"Error getting package list: {e}")
        
        st.write("### Environment Variables")
        env_vars = {k: v for k, v in os.environ.items() if not k.startswith("_")}
        st.code(str(env_vars))
        
        if st.button("Test Ollama Connection"):
            try:
                from langchain_ollama import ChatOllama
                model = ChatOllama(model="llama3")
                response = model.invoke("Hello, are you working?")
                st.success(f"Ollama test successful! Response: {response.content}")
            except Exception as e:
                st.error(f"Ollama test failed: {e}")
                st.code(traceback.format_exc())

def main():
    """Main function for the Streamlit app."""
    st.title("🤖 RAG Chatbot with Semantic Chunking, Reranking, and Query Routing")
    
    # Enable debug tab
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    
    # Initialize session state variables
    if "setup_complete" not in st.session_state:
        st.session_state.setup_complete = False
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar configuration
    with st.sidebar:
        st.title("Configuration")
        
        # Data directories and descriptions
        if "domain1_dir" not in st.session_state:
            st.session_state.domain1_dir = "./data/domain1"
        if "domain1_desc" not in st.session_state:
            st.session_state.domain1_desc = "First domain description"
            
        if "domain2_dir" not in st.session_state:
            st.session_state.domain2_dir = "./data/domain2"
        if "domain2_desc" not in st.session_state:
            st.session_state.domain2_desc = "Second domain description"
            
        if "persist_dir" not in st.session_state:
            st.session_state.persist_dir = "./chroma_db"
        
        if "model_name" not in st.session_state:
            st.session_state.model_name = "llama3"
        
        # Input fields
        st.session_state.domain1_dir = st.text_input("Domain 1 Directory", value=st.session_state.domain1_dir)
        st.session_state.domain1_desc = st.text_area("Domain 1 Description", value=st.session_state.domain1_desc)
        
        st.session_state.domain2_dir = st.text_input("Domain 2 Directory", value=st.session_state.domain2_dir)
        st.session_state.domain2_desc = st.text_area("Domain 2 Description", value=st.session_state.domain2_desc)
        
        st.session_state.persist_dir = st.text_input("Vector DB Directory", value=st.session_state.persist_dir)
        st.session_state.model_name = st.text_input("Ollama Model", value=st.session_state.model_name)
        
        # Setup button
        if not st.session_state.setup_complete:
            if st.button("Setup RAG Chatbot"):
                with st.spinner("Setting up RAG chatbot..."):
                    if setup_rag_chatbot():
                        st.rerun()  # Changed from experimental_rerun() to rerun()
        else:
            if st.button("Reset and Setup Again"):
                st.session_state.setup_complete = False
                if "chatbot" in st.session_state:
                    del st.session_state.chatbot
                st.rerun()  # Changed from experimental_rerun() to rerun()
        
        # Clear chat button
        if st.session_state.setup_complete and st.button("Clear Chat History"):
            st.session_state.messages = []
            if "chatbot" in st.session_state:
                st.session_state.chatbot.clear_conversation_history()
            st.rerun()  # Changed from experimental_rerun() to rerun()
            
        # Debug mode toggle
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
    
    # Display debug information if enabled
    if st.session_state.debug_mode:
        display_debug_info()
    
    # Main chat area
    if not st.session_state.setup_complete:
        st.info("Please configure and set up the RAG Chatbot using the sidebar before chatting.")
    else:
        # Display domain information
        st.write("### Knowledge Domains")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Domain 1**: {st.session_state.domain1_desc}")
        with col2:
            st.write(f"**Domain 2**: {st.session_state.domain2_desc}")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history and display
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    with st.spinner("Thinking..."):
                        response = st.session_state.chatbot.chat(prompt)
                        
                        # Check for empty response
                        if not response or not response.strip():
                            response = "I apologize, but I couldn't generate a proper response. Please try asking your question differently."
                        
                    message_placeholder.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error getting chatbot response: {error_msg}")
                    logger.error(traceback.format_exc())
                    message_placeholder.error(f"Error: {error_msg}. Please try again or reset the chatbot.")
                    
                    # Add error message to chat history
                    error_response = f"I encountered an error while processing your question. Error: {error_msg}"
                    st.session_state.messages.append({"role": "assistant", "content": error_response})

if __name__ == "__main__":
    main()