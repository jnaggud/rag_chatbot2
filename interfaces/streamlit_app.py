# interfaces/streamlit_app.py

import os
import sys
import subprocess
import platform
import traceback
import logging

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

from config.settings import (
    DEFAULT_PERSIST_DIRECTORY,
    DEFAULT_DESCRIPTIONS_FILE,
)
from config.settings import DEFAULT_PERSIST_DIRECTORY, DEFAULT_DESCRIPTIONS_FILE




logger = setup_logging()

st.set_page_config(
    page_title="🔗 RAG Chatbot",
    layout="wide",
)


def is_ollama_running() -> bool:
    # same logic as before...
    try:
        import requests
        r = requests.get("http://localhost:11434/api/version", timeout=2)
        return r.status_code == 200
    except Exception:
        cmd = "tasklist | findstr ollama" if platform.system()=="Windows" else "ps aux | grep '[o]llama'"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return bool(res.stdout.strip())


def init_rag():
    """One‐time setup: embeddings, chunking, vector store, LLM chain."""
    if not is_ollama_running():
        st.sidebar.error("❌ Ollama isn’t running. Run `ollama serve` first.")
        return False

    # read inputs from session_state
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
        info.text("1/5 Installing deps")
        progress.progress(10)
        check_and_install_dependencies()

        info.text("2/5 Checking Ollama")
        progress.progress(25)
        check_and_install_ollama()

        info.text(f"3/5 Pulling LLM: {model}")
        progress.progress(40)
        pull_ollama_model(model)

        info.text("4/5 Building embeddings & indexes")
        progress.progress(55)
        embed_fn = get_embedding_function()
        chunker  = SemanticChunkingProcessor(embed_fn)

        # ← pass descriptions_file here
        vdb      = VectorDatabaseManager(embed_fn, persist, desc_f)
        vdb.create_index("domain1", d1_desc)
        vdb.create_index("domain2", d2_desc)

        info.text("5/5 Chunking & indexing docs")
        progress.progress(70)
        docs1 = load_documents(d1_dir)
        if docs1:
            chunks1 = chunker.chunk_documents(docs1)
            vdb.add_documents_to_index("domain1", chunks1)

        docs2 = load_documents(d2_dir)
        if docs2:
            chunks2 = chunker.chunk_documents(docs2)
            vdb.add_documents_to_index("domain2", chunks2)

        info.text("Finalizing RAG chain")
        progress.progress(85)
        router    = QueryRouter(vdb, embed_fn)
        reranker  = SemanticReranker(embed_fn)
        retriever = RAGRetriever(vdb, router, reranker)
        chain     = RAGChain(retriever, model)
        chatbot   = RAGChatbot(chain)

        # persist into session_state
        st.session_state.chatbot        = chatbot
        st.session_state.setup_complete = True

        progress.progress(100, text="Setup complete 🎉")
        st.sidebar.success("RAG ready!")
        return True

    except Exception as e:
        st.sidebar.error(f"Setup failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def main():
    st.title("🔗 RAG Chatbot Configuration")

    # --- Sidebar inputs ---
    with st.sidebar:
        st.header("Settings")

        # Domain 1
        st.session_state.domain1_dir  = st.text_input(
            "Domain 1 data directory",
            value=st.session_state.get("domain1_dir","./data/domain1")
        )
        st.session_state.domain1_desc = st.text_area(
            "Domain 1 description",
            value=st.session_state.get("domain1_desc","First domain of documents")
        )

        # Domain 2
        st.session_state.domain2_dir  = st.text_input(
            "Domain 2 data directory",
            value=st.session_state.get("domain2_dir","./data/domain2")
        )
        st.session_state.domain2_desc = st.text_area(
            "Domain 2 description",
            value=st.session_state.get("domain2_desc","Second domain of documents")
        )

        # Chroma persistence & index descriptions file
        st.session_state.persist_dir       = st.text_input(
            "Chroma persist directory",
            value=st.session_state.get("persist_dir", DEFAULT_PERSIST_DIRECTORY)
        )
        st.session_state.descriptions_file = st.text_input(
            "Index descriptions file",
            value=st.session_state.get("descriptions_file", DEFAULT_DESCRIPTIONS_FILE)
        )

        # Model dropdown
        MODELS = ["llama3","nemotron"]
        default = st.session_state.get("model_name","llama3")
        if default not in MODELS:
            MODELS.insert(0, default)
        st.session_state.model_name = st.selectbox(
            "Choose LLM model",
            options=MODELS,
            index=MODELS.index(default),
        )

        st.markdown("---")
        if not st.session_state.get("setup_complete", False):
            if st.button("Setup RAG Chatbot"):
                with st.spinner("Initializing…"):
                    init_rag()
        else:
            if st.button("Re‑run setup"):
                st.session_state.setup_complete = False
                st.experimental_rerun()

    # --- Main chat area ---
    if not st.session_state.get("setup_complete", False):
        st.info("▶️ Click **Setup RAG Chatbot** to build your indexes first.")
        return

    st.header("💬 Chat with your documents")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show past
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input & response
    question = st.chat_input("Ask a question…")
    if question:
        st.session_state.messages.append({"role":"user","content":question})
        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Thinking…"):
                try:
                    answer = st.session_state.chatbot.chat(question)
                    if not answer.strip():
                        answer = "❗️ Empty response—try rephrasing."
                except Exception as e:
                    logger.error(traceback.format_exc())
                    answer = f"💥 Error: {str(e)}"
            placeholder.markdown(answer)
            st.session_state.messages.append({"role":"assistant","content":answer})


if __name__ == "__main__":
    main()
