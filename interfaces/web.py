"""
Web interface for the RAG chatbot.
"""

import logging

logger = logging.getLogger(__name__)

def create_web_interface(chatbot):
    """
    Create a simple web interface for the chatbot using FastAPI.
    
    Args:
        chatbot: The RAG chatbot instance
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        import uvicorn
        
        app = FastAPI(title="RAG Chatbot API")
        
        # Enable CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        class ChatRequest(BaseModel):
            message: str
        
        class ChatResponse(BaseModel):
            response: str
        
        @app.post("/chat", response_model=ChatResponse)
        async def chat_endpoint(request: ChatRequest):
            try:
                response = chatbot.chat(request.message)
                return ChatResponse(response=response)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/history")
        async def get_history():
            return chatbot.get_conversation_history()
        
        @app.post("/clear")
        async def clear_history():
            chatbot.clear_conversation_history()
            return {"status": "History cleared"}
        
        # Run the server
        logger.info("Starting web interface on http://0.0.0.0:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError as e:
        logger.error(f"Missing dependencies for web interface: {e}")
        logger.info("Please install fastapi and uvicorn: pip install fastapi uvicorn")
