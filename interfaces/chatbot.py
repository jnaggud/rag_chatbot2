"""
Core chatbot interface for the RAG system.
"""

import logging
import traceback
from typing import List, Dict

logger = logging.getLogger(__name__)

class RAGChatbot:
    """
    Simple chatbot interface for the RAG system.
    """
    
    def __init__(self, rag_chain):
        """
        Initialize the RAG chatbot.
        
        Args:
            rag_chain: The RAG chain
        """
        self.rag_chain = rag_chain
        self.conversation_history = []
        
        logger.info("RAG Chatbot initialized.")
    
    def chat(self, user_input: str) -> str:
        """
        Process a user input and generate a response.
        
        Args:
            user_input: The user's input
            
        Returns:
            Generated response
        """
        if not user_input or not user_input.strip():
            return "Please ask a question."
            
        try:
            # Log the user input
            logger.info(f"User input: {user_input}")
            
            # Add user input to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Generate response using RAG chain
            response = self.rag_chain.run(user_input)
            
            # Check if response is empty and provide fallback
            if not response or not response.strip():
                logger.warning("Received empty response from RAG chain")
                response = "I'm sorry, I couldn't generate a proper response. Please try rephrasing your question."
            
            logger.info(f"Generated response: {response[:100]}...")
            
            # Add response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in chatbot: {error_msg}")
            logger.error(traceback.format_exc())
            
            # Create user-friendly error message
            friendly_error = f"I'm sorry, I encountered an error while processing your question. Error: {error_msg}"
            
            # Add error message to conversation history
            self.conversation_history.append({"role": "assistant", "content": friendly_error})
            
            return friendly_error
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of conversation turns
        """
        return self.conversation_history
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared.")