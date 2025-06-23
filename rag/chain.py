# rag/chain.py

import logging, traceback
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from config.settings import DEFAULT_LLM_MODEL

logger = logging.getLogger(__name__)

class RAGChain:
    """
    RAG chain that retrieves, reranks, then prompts a generic expert assistant.
    """
    def __init__(self, retriever, model_name: str = DEFAULT_LLM_MODEL):
        self.retriever = retriever
        self.model_name = model_name

        # ChatOllama setup
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.3,
            num_ctx=4096,
            repeat_penalty=1.1,
            top_k=10,
            top_p=0.95
        )

        # **Generic expert assistant prompt**
        self.prompt = PromptTemplate.from_template(
            """
You are a highly knowledgeable expert assistant. Use *only* the provided context to answer the user's question combined with your expert knowledge.

## Instructions:
1. Provide a clear, concise answer to the question.
2. If there are procedural steps, present them as bullet points.
3. Include a summary paragraph that synthesizes the answer.
4. **Always cite your sources** using the format [Source: filename, page X] where X is the page number.
5. If a document doesn't specify a page, use [Source: filename].
6. If multiple sources support the same information, list them all: [Source: doc1.pdf, page 1; doc2.pdf, page 3]
7. If the context doesn't contain relevant information, say "I don't have enough information to answer this question."

## Context:
{context}

## Question: {question}

## Answer:
"""
        )

        self._build_chain()
        logger.info(f"RAGChain initialized with model {model_name}")

    def _build_chain(self):
        try:
            # run retrieval in parallel, pass question straight through
            runnable_map = RunnableParallel(
                context=lambda q: self.retriever.retrieve(q),
                question=RunnablePassthrough()
            )
            self.chain = (
                runnable_map
                | (lambda x: {
                    "context": self._format_docs(x["context"]), 
                    "question": x["question"],
                    "sources": [doc.metadata for doc in x["context"]] if hasattr(x["context"][0], "metadata") 
                              else [doc.get("metadata", {}) for doc in x["context"]]
                })
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        except Exception:
            logger.error("Failed to build chain", exc_info=True)
            raise

    def _format_docs(self, docs):
        """
        Format retrieved documents (now dicts) into a single context string with source metadata.

        Args:
            docs: List[dict] each with keys "page_content" & "metadata"

        Returns:
            A formatted string of all document texts with source information.
        """
        if not docs:
            return "No relevant information found."

        formatted = []
        for i, doc in enumerate(docs):
            # Extract content and metadata
            if hasattr(doc, "page_content"):
                text = doc.page_content
                metadata = getattr(doc, "metadata", {})
            else:
                text = doc.get("page_content", "")
                metadata = doc.get("metadata", {})
            
            # Get source information
            source = metadata.get('source', 'Unknown source')
            page = metadata.get('page', 'N/A')
            
            # Format with source information
            source_info = f"[Source: {source}"
            if page != 'N/A':
                source_info += f", page {page}"
            source_info += "]"
            
            formatted.append(f"Document {i+1} ({source_info}):\n{text}")

        return "\n\n".join(formatted)


    def run(self, query: str) -> str:
        if not query.strip():
            return "Please ask a question."
        try:
            logger.info(f"Invoking chain on query: {query}")
            resp = self.chain.invoke(query)
            return resp.strip() or "I’m sorry, I couldn’t generate a response. Please try again."
        except Exception as e:
            logger.error("Error in RAGChain.run", exc_info=True)
            return f"I encountered an error: {e}"
