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
You are a highly knowledgeable expert assistant. Use *only* the provided context to answer the user’s question combined with your expert knowledge.

- If there are procedural steps, present them as **clear bullet points**.
- After bullets, provide a **concise summary paragraphs** that synthesizes the answer.
- If the context does not contain the information, respond with “I don’t have enough information to answer this.”

Context:
{context}

Question: {question}

Answer:
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
                | (lambda x: {"context": self._format_docs(x["context"]), "question": x["question"]})
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        except Exception:
            logger.error("Failed to build chain", exc_info=True)
            raise

    def _format_docs(self, docs):
        if not docs:
            return "No relevant information found."
        return "\n\n".join(f"Document {i+1}:\n{d.page_content}" for i, d in enumerate(docs))

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
