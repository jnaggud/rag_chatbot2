# rag/chunking.py

import logging
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter
)
from typing import List, Dict, Any
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class SemanticChunkingProcessor:
    """
    Enhanced chunking processor that handles different types of content:
    - For markdown/structured text: Uses heading-aware splitting
    - For code: Uses language-specific splitting
    - Fallback: Uses improved recursive character splitting
    """
    
    def __init__(self):
        # Main splitter for general text
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n## ",  # Major headings
                "\n## ",    # Major headings (no leading newline)
                "\n\n# ",   # Main headings
                "\n# ",      # Main headings (no leading newline)
                "\n\n",      # Paragraphs
                ". ", "! ", "? ",  # Sentences
                " ",         # Words (last resort)
                ""
            ],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        
        # Markdown splitter for structured content
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("##", "Section"),
                ("###", "Subsection"),
                ("####", "Subsubsection"),
            ],
            return_each_line=False,
        )
        
        # Code splitter for code blocks
        self.code_splitter = PythonCodeTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        logger.info(f"Chunker configured with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    def _is_markdown(self, text: str) -> bool:
        """Check if text contains markdown formatting."""
        markdown_patterns = ["# ", "## ", "### ", "#### ", "- ", "* ", "```"]
        return any(pattern in text for pattern in markdown_patterns)

    def _is_code(self, text: str) -> bool:
        """Check if text appears to be code."""
        code_indicators = ["def ", "class ", "import ", "from ", "return "]
        return any(indicator in text for indicator in code_indicators)

    def _get_doc_content(self, doc):
        """Safely get content from either a dictionary or Document object."""
        if hasattr(doc, 'page_content'):  # Handle Document objects
            return doc.page_content, doc.metadata if hasattr(doc, 'metadata') else {}
        # Handle dictionary-style objects
        return doc.get('page_content', ''), doc.get('metadata', {})

    def chunk_documents(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """Process a list of documents with appropriate chunking strategy."""
        all_chunks = []
        
        for doc in documents:
            content, metadata = self._get_doc_content(doc)
            
            try:
                if self._is_markdown(content):
                    # Use markdown splitter for structured content
                    chunks = self.markdown_splitter.split_text(content)
                    for chunk in chunks:
                        if hasattr(chunk, 'metadata'):  # If it's a Document
                            chunk.metadata.update(metadata)
                            all_chunks.append(chunk)
                        else:  # If it's a string
                            all_chunks.append({
                                'page_content': chunk,
                                'metadata': metadata
                            })
                elif self._is_code(content):
                    # Use code splitter for code
                    chunks = self.code_splitter.split_text(content)
                    for chunk in chunks:
                        all_chunks.append({
                            'page_content': chunk,
                            'metadata': metadata
                        })
                else:
                    # Use general text splitter
                    if hasattr(doc, 'page_content'):  # If it's a Document
                        chunks = self.text_splitter.split_documents([doc])
                    else:  # If it's a dict
                        chunks = self.text_splitter.split_documents([{
                            'page_content': content,
                            'metadata': metadata
                        }])
                    all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Error processing document: {e}")
                # Fallback to simple split
                try:
                    if hasattr(doc, 'page_content'):
                        chunks = self.text_splitter.split_documents([doc])
                    else:
                        chunks = self.text_splitter.split_documents([{
                            'page_content': content,
                            'metadata': metadata
                        }])
                    all_chunks.extend(chunks)
                except Exception as e2:
                    logger.error(f"Fallback splitting also failed: {e2}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            if hasattr(chunk, 'page_content'):  # If it's a Document
                content = chunk.page_content
                chunk_dict = {
                    'page_content': content,
                    'metadata': chunk.metadata if hasattr(chunk, 'metadata') else {}
                }
            else:  # If it's a dict
                content = chunk.get('page_content', '')
                chunk_dict = chunk
            
            # Skip empty or very short chunks
            if not content or len(str(content).strip()) < 10:
                continue
                
            # Create a unique identifier for this chunk
            chunk_meta = chunk_dict.get('metadata', {})
            source = chunk_meta.get('source', '')
            page = chunk_meta.get('page', '')
            chunk_id = f"{source}:{page}:{str(content)[:100]}"
            
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_chunks.append(chunk_dict)
        
        logger.info(f"Split {len(documents)} documents into {len(unique_chunks)} chunks")
        return unique_chunks
