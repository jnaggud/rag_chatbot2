"""
RAG chain implementation for the RAG chatbot.
"""

import logging
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from config.settings import DEFAULT_LLM_MODEL
import traceback

logger = logging.getLogger(__name__)

class RAGChain:
    """
    Complete RAG chain that combines retrieval with generation.
    """
    
    def __init__(
        self,
        retriever,
        model_name: str = DEFAULT_LLM_MODEL
    ):
        """
        Initialize the RAG chain.
        
        Args:
            retriever: Document retriever
            model_name: Name of the LLM model
        """
        self.retriever = retriever
        self.model_name = model_name
        
        # Initialize LLM - removed format="json" which can cause issues
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.7,
            num_ctx=4096,  # Increase context window
            repeat_penalty=1.1,
            top_k=10,
            top_p=0.95,
            max_tokens=2048,
            stream=True
        )
        
        # Initialize the RAG prompt with clearer instructions
        self.prompt = PromptTemplate.from_template(
                """
            You are a highly knowledgeable research assistant.
            Answer the user’s question using *only* the provided context.
            If the answer is not present, say “I don’t have enough information.”

            Before answering, think through each step and cite every fact with its document number, e.g. [Doc 2].

            Provide a **detailed**, **step‑by‑step** explanation of **at least 500 words**, covering every nuance and assumption.
            Use paragraphs and subheadings as needed; do not use bullet points unless it clarifies the structure.

            Context:
            {context}

            Question: {question}

            Answer:
            """
        )
        
        # Build the RAG chain
        self.setup_chain()
        
        logger.info(f"RAG Chain initialized with model {model_name}.")
    
    def setup_chain(self):
        """Set up the RAG chain with proper error handling."""
        try:
            # Create a RunnableParallel for the context and question
            runnable_map = RunnableParallel(
                context=lambda x: self.retriever.retrieve(x),
                question=RunnablePassthrough()
            )
            
            # Build the complete chain
            self.chain = (
                runnable_map 
                | (lambda x: {"context": self._format_docs(x["context"]), "question": x["question"]})
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
        except Exception as e:
            logger.error(f"Failed to set up RAG chain: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _format_docs(self, docs):
        """
        Format retrieved documents into a single context string.
        
        Args:
            docs: List of retrieved documents
            
        Returns:
            A formatted string of document contents
        """
        if not docs:
            return "No relevant information found."
        
        formatted_docs = []
        for i, doc in enumerate(docs):
            formatted_docs.append(f"Document {i+1}:\n{doc.page_content}")
        
        return "\n\n".join(formatted_docs)
    
    def run(self, query: str) -> str:
        """
        Run the RAG chain on a query.
        
        Args:
            query: The user query
            
        Returns:
            Generated response
        """
        if not query or not query.strip():
            return "Please ask a question."
            
        try:
            logger.info(f"Running RAG chain for query: '{query}'")
            
            # Add a fallback response in case of an empty result
            response = self.chain.invoke(query)
            
            # Check if response is empty and provide a fallback
            if not response or not response.strip():
                logger.warning("Received empty response from LLM")
                return "I'm sorry, I couldn't generate a proper response. Please try rephrasing your question."
                
            return response
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error running RAG chain: {error_msg}")
            logger.error(traceback.format_exc())
            
            # Return a user-friendly error message
            return f"I encountered an error while processing your query. Please try again or ask a different question. Error details: {error_msg}"