# interfaces/streamlit_app.py

import os
import sys
import subprocess
import platform
import traceback
import logging
import json
import plotly.express as px
from utils.metrics import metrics_collector, QueryStatus
from pathlib import Path
from datetime import datetime
import time

import streamlit as st

# bring project root onto PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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

from config.settings import DEFAULT_PERSIST_DIRECTORY, DEFAULT_DESCRIPTIONS_FILE, AVAILABLE_LLM_MODELS, DEFAULT_LLM_MODEL

logger = setup_logging()

# Set page config
st.set_page_config(
    page_title="🔗 RAG Chatbot",
    layout="wide",
    page_icon="🔍"
)

# Page navigation
PAGES = {
    "Chat": "chat",
    "Observability Dashboard": "dashboard"
}

def show_dashboard():
    """Display the observability dashboard."""
    st.title("🔍 RAG System Observability")
    
    # Get metrics data
    df = metrics_collector.get_metrics_dataframe()
    
    if df.empty:
        st.info("No query metrics available yet. Use the chat to generate some data.")
        return
    
    # Summary stats
    st.header("📊 Summary Statistics")
    stats = metrics_collector.get_summary_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Queries", stats['total_queries'])
    with col2:
        st.metric("Success Rate", f"{stats['success_rate']*100:.1f}%")
    with col3:
        st.metric("Avg Latency", f"{stats['avg_latency']:.2f}s")
    with col4:
        st.metric("Total Tokens Used", f"{stats['total_tokens_used']:,}")
    
    # Token usage over time
    st.header("📈 Token Usage Over Time")
    if len(df) > 1:
        fig = px.line(df, x='timestamp', y=['input_tokens', 'output_tokens', 'total_tokens'],
                     title="Token Usage Over Time",
                     labels={'value': 'Tokens', 'variable': 'Token Type'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Latency distribution
    st.header("⏱️ Latency Distribution")
    if len(df) > 1:
        fig = px.histogram(df, x='latency_seconds', nbins=20,
                         title="Query Latency Distribution",
                         labels={'latency_seconds': 'Latency (seconds)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Most recent queries
    st.header("📝 Recent Queries")
    st.dataframe(df[['timestamp', 'query', 'status', 'latency_seconds', 'total_tokens']]
                 .sort_values('timestamp', ascending=False)
                 .head(10),
                 use_container_width=True)
    
    # Chunk sources
    st.header("📚 Top Document Sources")
    if not df.empty and 'chunk_sources' in df.columns and not df['chunk_sources'].isna().all():
        sources = df.explode('chunk_sources')['chunk_sources'].value_counts().reset_index()
        sources.columns = ['Source', 'Count']
        if not sources.empty:
            fig = px.bar(sources.head(10), x='Source', y='Count',
                        title="Most Used Document Sources")
            st.plotly_chart(fig, use_container_width=True)
    
    # Raw data export
    st.header("📥 Export Data")
    if st.button("Export Metrics to CSV"):
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"rag_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
        )

def is_ollama_running() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/version", timeout=2)
        return r.status_code == 200
    except Exception:
        cmd = "tasklist | findstr ollama" if platform.system()=="Windows" else "ps aux | grep '[o]llama'"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return bool(res.stdout.strip())


def init_rag(force_rebuild: bool):
    """
    Set up embeddings, vector store, chunking, and the RAG chain.
    If force_rebuild is False and indexes already exist, we skip chunking/indexing.
    """
    if not is_ollama_running():
        st.sidebar.error("❌ Ollama isn’t running. Run `ollama serve` first.")
        return False

    # Read sidebar inputs
    d1_dir  = st.session_state.domain1_dir
    d2_dir  = st.session_state.domain2_dir
    d1_desc = st.session_state.domain1_desc
    d2_desc = st.session_state.domain2_desc
    persist = st.session_state.persist_dir
    desc_f  = st.session_state.descriptions_file
    model   = st.session_state.model_name

    progress = st.sidebar.progress(0, text="Starting setup…")
    info     = st.sidebar.empty()

    try:
        # 1. Dependencies
        info.text("1/6 Installing dependencies")
        progress.progress(10)
        check_and_install_dependencies()

        # 2. Ollama
        info.text("2/6 Checking Ollama")
        progress.progress(25)
        check_and_install_ollama()

        # 3. Pull model
        info.text(f"3/6 Pulling LLM: {model}")
        progress.progress(40)
        pull_ollama_model(model)

        # 4. Embeddings + Chunker + VectorStore
        info.text("4/6 Building embedding function")
        progress.progress(50)
        embed_fn = get_embedding_function()

        chunker = SemanticChunkingProcessor()
        vdb     = VectorDatabaseManager(embed_fn, persist, desc_f)

        # ---- FIXED: simply get a set of existing names ----
        existing_cols = set(vdb.client.list_collections())

        needs_rebuild = force_rebuild or not ({"domain1", "domain2"} <= existing_cols)

        if needs_rebuild:
            info.text("5/6 (Re)creating indexes & descriptions")
            progress.progress(60)
            vdb.create_index("domain1", d1_desc)
            vdb.create_index("domain2", d2_desc)

            info.text("6/6 Chunking & indexing documents")
            progress.progress(70)

            # Pass enable_ocr to load_documents
            enable_ocr = st.session_state.get("enable_ocr", False)
            
            docs1 = load_documents(d1_dir, enable_ocr=enable_ocr)
            if docs1:
                chunks1 = chunker.chunk_documents(docs1)
                vdb.add_documents_to_index("domain1", chunks1)

            docs2 = load_documents(d2_dir, enable_ocr=enable_ocr)
            if docs2:
                chunks2 = chunker.chunk_documents(docs2)
                vdb.add_documents_to_index("domain2", chunks2)
        else:
            info.text("✅ Using existing indexes (no rebuild)")
            progress.progress(70)

        # 5. Build the RAG chain
        info.text("Finalizing RAG chain")
        progress.progress(85)
        router    = QueryRouter(vdb, embed_fn)
        reranker  = SemanticReranker(embed_fn)
        retriever = RAGRetriever(vdb, router, reranker)
        chain     = RAGChain(retriever, model)
        chatbot   = RAGChatbot(chain)

        # Persist into state
        st.session_state.chatbot        = chatbot
        st.session_state.setup_complete = True

        progress.progress(100, text="Setup complete 🎉")
        st.sidebar.success("RAG ready!")
        return True

    except Exception as e:
        st.sidebar.error(f"Setup failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def chat_page():
    """Main chat interface."""
    st.title("🔗 RAG Chatbot Configuration & Chat")
    
    # --- Initialize session state for settings if they aren't already set ---
    # This section runs on every script rerun, but conditions like 
    # `if "key" not in st.session_state:` ensure initialization happens once.

    # 1. Descriptions file path
    if "descriptions_file" not in st.session_state:
        st.session_state.descriptions_file = DEFAULT_DESCRIPTIONS_FILE
    descriptions_filepath = st.session_state.descriptions_file

    # 2. Domain descriptions (load from file or use fallbacks)
    default_desc1 = "First domain of documents"
    default_desc2 = "Second domain of documents"
    
    loaded_desc1 = default_desc1
    loaded_desc2 = default_desc2

    if os.path.exists(descriptions_filepath):
        try:
            with open(descriptions_filepath, "r") as f:
                descriptions_data = json.load(f)
                loaded_desc1 = descriptions_data.get("domain1", default_desc1)
                loaded_desc2 = descriptions_data.get("domain2", default_desc2)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load or parse {descriptions_filepath}: {e}. Using default descriptions.")
    
    # Initialize session state for text_areas if not already set by user interaction
    if "domain1_desc" not in st.session_state:
        st.session_state.domain1_desc = loaded_desc1
    if "domain2_desc" not in st.session_state:
        st.session_state.domain2_desc = loaded_desc2

    # 3. Other settings paths and model (initialize if not in session state)
    if "domain1_dir" not in st.session_state:
        st.session_state.domain1_dir = "./data/domain1"
    if "domain2_dir" not in st.session_state:
        st.session_state.domain2_dir = "./data/domain2"
    if "persist_dir" not in st.session_state:
        st.session_state.persist_dir = DEFAULT_PERSIST_DIRECTORY
    if "model_name" not in st.session_state:
        st.session_state.model_name = DEFAULT_LLM_MODEL
    if "enable_ocr" not in st.session_state:
        st.session_state.enable_ocr = False
    
    # Initialize chatbot in session_state if not present
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None


    # --- Sidebar UI for Configuration ---
    with st.sidebar:
        st.header("Settings")

        st.text_input("Domain 1 data directory", key="domain1_dir")
        st.text_area("Domain 1 description", key="domain1_desc", height=100)

        st.text_input("Domain 2 data directory", key="domain2_dir")
        st.text_area("Domain 2 description", key="domain2_desc", height=100)

        st.text_input("Chroma persist directory", key="persist_dir")
        st.text_input("Index descriptions file", key="descriptions_file")
        
        # Add OCR toggle
        st.checkbox("Enable OCR for PDF images", key="enable_ocr", value=False,
                  help="Enable OCR to extract text from images in PDFs (slower but more accurate for scanned documents)")
        
        current_model_in_state = st.session_state.model_name
        if current_model_in_state not in AVAILABLE_LLM_MODELS:
            logger.warning(f"Model '{current_model_in_state}' in session state not in AVAILABLE_LLM_MODELS. Defaulting to first available.")
            st.session_state.model_name = AVAILABLE_LLM_MODELS[0] if AVAILABLE_LLM_MODELS else ""

        st.selectbox(
            "Select LLM Model",
            options=AVAILABLE_LLM_MODELS, # Ensure AVAILABLE_LLM_MODELS is loaded from config
            key="model_name"
        )

        if st.button("🚀 Initialize/Rebuild RAG System", type="primary"):
            if not is_ollama_running():
                 st.sidebar.error("❌ Ollama isn’t running. Run `ollama serve` first.")
            else:
                with st.spinner("Initializing RAG system..."):
                    success = init_rag(force_rebuild=True) 
                    if success:
                        # init_rag now correctly sets st.session_state.chatbot
                        st.sidebar.success("RAG system initialized/rebuilt successfully!")
                    else:
                        st.sidebar.error("RAG system initialization/rebuild failed. Check logs.")
        
        if st.button("🔄 Reload Descriptions from File"):
            descriptions_filepath_reload = st.session_state.descriptions_file
            reloaded_desc1_btn = default_desc1
            reloaded_desc2_btn = default_desc2
            if os.path.exists(descriptions_filepath_reload):
                try:
                    with open(descriptions_filepath_reload, "r") as f:
                        descriptions_data = json.load(f)
                        reloaded_desc1_btn = descriptions_data.get("domain1", default_desc1)
                        reloaded_desc2_btn = descriptions_data.get("domain2", default_desc2)
                    st.session_state.domain1_desc = reloaded_desc1_btn
                    st.session_state.domain2_desc = reloaded_desc2_btn
                    st.sidebar.success(f"Descriptions reloaded into UI from {descriptions_filepath_reload}")
                    st.experimental_rerun() 
                except (json.JSONDecodeError, IOError) as e:
                    st.sidebar.error(f"Failed to reload from {descriptions_filepath_reload}: {e}")
            else:
                st.sidebar.warning(f"File not found: {descriptions_filepath_reload}")

    # --- Main Chat Interface ---
    st.header("💬 Chat with your RAG System")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you today? (Please initialize RAG system from sidebar if you haven't yet)"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            if st.session_state.chatbot is None:
                if not is_ollama_running():
                    full_response = "Ollama is not running. Please start Ollama and initialize the RAG system from the sidebar."
                else:
                    full_response = "RAG system is not initialized. Please use the 'Initialize/Rebuild RAG System' button in the sidebar."
                st.warning(full_response)
            else:
                # Start tracking the query
                query_id = metrics_collector.start_query(
                    query=prompt,
                    model=st.session_state.model_name,
                    rag_initialized='chatbot' in st.session_state
                )
                
                try:
                    with st.spinner("Thinking..."):
                        # Get the response from the chatbot
                        start_time = time.time()
                        response = st.session_state.chatbot.chat(prompt)
                        processing_time = time.time() - start_time
                        
                        # Get retrieved chunks if available
                        retrieved_chunks = []
                        chunk_scores = []
                        if hasattr(st.session_state.chatbot.rag_chain.retriever, 'last_retrieved_docs'):
                            retrieved_chunks = [
                                {'page_content': doc['page_content'], 'metadata': doc.get('metadata', {})}
                                for doc in st.session_state.chatbot.rag_chain.retriever.last_retrieved_docs
                                if isinstance(doc, dict) and 'page_content' in doc
                            ]
                            chunk_scores = getattr(st.session_state.chatbot.rag_chain.retriever, 'last_scores', [])
                        
                        # Complete the query tracking
                        metrics_collector.complete_query(
                            query_id=query_id,
                            response=response,
                            status=QueryStatus.SUCCESS,
                            retrieved_chunks=retrieved_chunks,
                            chunk_scores=chunk_scores,
                            processing_time_seconds=processing_time
                        )
                        
                        # Display the response
                        full_response = response
                        message_placeholder.markdown(full_response)
                        
                except Exception as e:
                    # Log the error
                    error_msg = str(e)
                    logger.error(f"Error in chat: {error_msg}", exc_info=True)
                    
                    # Update metrics with error
                    metrics_collector.complete_query(
                        query_id=query_id,
                        response="",
                        status="error",
                        error_message=error_msg
                    )
                    
                    # Show error to user
                    full_response = f"An error occurred: {error_msg}"
                    message_placeholder.error(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    
    # Initialize session state for metrics if not exists
    if 'metrics_initialized' not in st.session_state:
        metrics_file = Path("metrics.json")
        if metrics_file.exists():
            metrics_collector.load_from_file(metrics_file)
        st.session_state.metrics_initialized = True
    
    # Save metrics periodically
    if len(metrics_collector.queries) > 0:
        metrics_file = Path("metrics.json")
        metrics_collector.save_to_file(metrics_file)
    
    # Show the selected page
    if PAGES[selection] == "dashboard":
        show_dashboard()
    else:
        chat_page()

if __name__ == "__main__":
    main()
