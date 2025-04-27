# utils/hyde_utils.py

from langchain_ollama import ChatOllama
from rag.embeddings import get_embedding_function
from config.settings import DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL

def generate_hypothetical_answer(query, model_name=DEFAULT_LLM_MODEL):
    llm = ChatOllama(model=model_name, temperature=0.7)
    prompt = f"Please write a plausible answer to the following question as if you were an expert:\n\nQuestion: {query}\nAnswer:"
    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        return response.content.strip()
    return str(response).strip()

def hyde_embed_query(query, model_name=DEFAULT_LLM_MODEL, embedding_model=DEFAULT_EMBEDDING_MODEL):
    hypothetical_answer = generate_hypothetical_answer(query, model_name)
    embed_fn = get_embedding_function(embedding_model)
    embedding = embed_fn.embed_query(hypothetical_answer)
    return embedding