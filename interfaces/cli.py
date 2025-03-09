"""
Command-line interface for the RAG chatbot.
"""

import logging

logger = logging.getLogger(__name__)

def run_cli(chatbot):
    """
    Run a simple command-line interface for the chatbot.
    
    Args:
        chatbot: The RAG chatbot instance
    """
    print("\n===== RAG Chatbot with Semantic Chunking, Reranking, and Query Routing =====")
    print("Type 'exit' or 'quit' to end the conversation.")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        response = chatbot.chat(user_input)
        print(f"\nChatbot: {response}")
