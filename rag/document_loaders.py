# rag/document_loaders.py

import os
import glob
from langchain.document_loaders import PyPDFLoader

def load_documents(directory: str):
    """
    Recursively load all PDFs under `directory`, split them per page,
    and tag each Document with its source filepath.
    """
    documents = []
    # Find every PDF in subfolders
    pattern = os.path.join(directory, "**", "*.pdf")
    for path in glob.glob(pattern, recursive=True):
        # 1) Initialize loader with path only
        loader = PyPDFLoader(path)
        # 2) Split into page-level Documents
        docs = loader.load_and_split()
        # 3) Attach the source filename to metadata
        for doc in docs:
            # ensure there's a metadata dict
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}
            doc.metadata["source"] = path
        documents.extend(docs)
    return documents
