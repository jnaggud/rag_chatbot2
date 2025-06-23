# rag/document_loaders.py

import os
import re
import logging
import fitz  # PyMuPDF
import numpy as np
import pytesseract
import time
from typing import List, Dict, Any, Optional, Tuple, Union, Generator
from pathlib import Path
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime
from pypdf import PdfReader
from PIL import Image
import tiktoken
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader
import concurrent.futures

logger = logging.getLogger(__name__)

@dataclass
class PDFElement:
    """Represents a logical element in a PDF with its position and type."""
    text: str
    page_num: int
    element_type: str  # 'text', 'table', 'figure', 'header', 'footer'
    bbox: Optional[Tuple[float, float, float, float]] = None
    metadata: Optional[Dict[str, Any]] = None

class EnhancedPDFProcessor:
    """Enhanced PDF processing with layout analysis and content extraction."""
    
    def __init__(self, extract_tables: bool = True, extract_figures: bool = True, 
                 enable_ocr: bool = False, max_workers: int = None):
        self.extract_tables = extract_tables
        self.extract_figures = extract_figures
        self.enable_ocr = enable_ocr
        self.figure_count = 0
        self.max_workers = max_workers or os.cpu_count()
        
    def process_pdf(self, file_path: str) -> List[PDFElement]:
        """Process a PDF file and extract structured content with parallel page processing."""
        try:
            with fitz.open(file_path) as doc:
                # Process pages in parallel
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    # Process pages in batches to avoid memory issues
                    page_batches = [
                        (i, doc.load_page(i)) 
                        for i in range(len(doc))
                    ]
                    
                    # Process batch of pages in parallel
                    future_to_page = {
                        executor.submit(self._process_page, page, page_num): page_num
                        for page_num, page in page_batches
                    }
                    
                    # Process results as they complete
                    elements = []
                    for future in concurrent.futures.as_completed(future_to_page):
                        try:
                            page_elements = future.result()
                            elements.extend(page_elements)
                        except Exception as e:
                            page_num = future_to_page[future]
                            logger.error(f"Error processing page {page_num + 1}: {e}")
                    
                    return elements
                    
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise

    def _process_page(self, page: fitz.Page, page_num: int) -> List[PDFElement]:
        """Process a single PDF page and extract elements."""
        elements = []
        
        try:
            text_blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)["blocks"]
            
            # Process blocks in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for block in text_blocks:
                    if "image" in block and self.extract_figures:
                        futures.append(executor.submit(
                            self._process_image_block, block, page, page_num
                        ))
                    elif "lines" in block:
                        futures.append(executor.submit(
                            self._process_text_block, block, page, page_num
                        ))
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if isinstance(result, list):
                            elements.extend(result)
                        elif result is not None:
                            elements.append(result)
                    except Exception as e:
                        logger.error(f"Error processing block: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {e}")
                
        return elements
    
    def _process_text_block(self, block: Dict, page: fitz.Page, page_num: int) -> List[PDFElement]:
        """Process a text block and classify it."""
        elements = []
        try:
            text = "\n".join(" ".join(span["text"] for span in line["spans"] if span["text"].strip()) 
                             for line in block.get("lines", []) 
                             if any(span["text"].strip() for span in line.get("spans", [])))
            
            if not text.strip():
                return elements
                
            bbox = block["bbox"]
            
            # Classify text block
            if self._is_header_footer(block, page.rect):
                element_type = "header" if bbox[1] < page.rect.height * 0.2 else "footer"
            else:
                element_type = "text"
                
            elements.append(PDFElement(
                text=text,
                page_num=page_num + 1,  # 1-based page numbering
                element_type=element_type,
                bbox=bbox,
                metadata={
                    "font_sizes": list({span["size"] for line in block.get("lines", []) 
                                      for span in line.get("spans", []) if "size" in span}),
                    "is_bold": any(span.get("font", "").lower().find("bold") >= 0 
                                 for line in block.get("lines", []) 
                                 for span in line.get("spans", [])),
                    "is_italic": any(span.get("font", "").lower().find("italic") >= 0 
                                   for line in block.get("lines", []) 
                                   for span in line.get("spans", []))
                }
            ))
        except Exception as e:
            logger.error(f"Error processing text block: {e}")
            
        return elements
    
    def _process_image_block(self, block: Dict, page: fitz.Page, page_num: int) -> PDFElement:
        """Process an image block, optionally using OCR if enabled."""
        self.figure_count += 1
        image_bytes = block["image"]
        img = Image.open(BytesIO(image_bytes))
        
        # Only use OCR if enabled and the image is not just a background/decoration
        text = f"[Figure {self.figure_count}]"  # Default text
        
        if self.enable_ocr:
            try:
                # Check if this is likely a real image (not just a background/decoration)
                # by checking if it has significant non-white content
                if self._is_real_image(img):
                    ocr_text = pytesseract.image_to_string(img).strip()
                    if ocr_text:
                        text = f"[Figure {self.figure_count}]\n{ocr_text}"
            except Exception as e:
                logger.warning(f"OCR failed for image on page {page_num + 1}: {e}")
                    
        return PDFElement(
            text=text,
            page_num=page_num + 1,
            element_type="figure",
            bbox=block["bbox"],
            metadata={
                "figure_number": self.figure_count,
                "dimensions": img.size,
                "format": img.format,
                "ocr_used": self.enable_ocr and self._is_real_image(img)
            }
        )
    
    @staticmethod
    def _is_header_footer(block: Dict, page_rect) -> bool:
        """Determine if a block is a header or footer."""
        bbox = block["bbox"]
        # Check if block is in top or bottom 10% of the page
        return (bbox[1] < page_rect.height * 0.1 or 
                bbox[3] > page_rect.height * 0.9)
    
    @staticmethod
    def _is_real_image(img: Image.Image, threshold: float = 0.95) -> bool:
        """
        Check if an image is likely to contain meaningful content.
        Returns False for mostly white/blank images or small images.
        """
        # Skip very small images (likely icons or decoration)
        if img.width < 50 or img.height < 50:
            return False
            
        # Convert to grayscale and count non-white pixels
        gray = img.convert('L')
        pixels = list(gray.getdata())
        
        # Count non-white pixels (assuming white is 255)
        non_white = sum(1 for p in pixels if p < 250)
        ratio = non_white / len(pixels)
        
        # Consider it a real image if more than 5% of pixels are non-white
        return ratio > 0.05
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean extracted text from PDF."""
        if not text:
            return ""
            
        # Remove multiple spaces, newlines, etc.
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove common PDF artifacts
        text = re.sub(r'\n', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
        return text

class PDFChunkingStrategy:
    """Specialized chunking strategy for PDF documents."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
    def chunk_elements(self, elements: List[PDFElement]) -> List[Dict[str, Any]]:
        """Chunk PDF elements while preserving logical structure."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for element in elements:
            element_text = element.text.strip()
            if not element_text:
                continue
                
            element_tokens = len(self.encoder.encode(element_text))
            
            # Start new chunk if adding this element would exceed chunk size
            if current_chunk and current_size + element_tokens > self.chunk_size:
                chunks.append(self._finalize_chunk(current_chunk))
                # Keep some overlap
                current_chunk = current_chunk[-2:] if len(current_chunk) > 2 else []
                current_size = sum(len(self.encoder.encode(el.text)) for el in current_chunk)
                
            current_chunk.append(element)
            current_size += element_tokens
            
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(self._finalize_chunk(current_chunk))
            
        return chunks
    
    def _finalize_chunk(self, elements: List[PDFElement]) -> Dict[str, Any]:
        """Convert a list of elements into a chunk with metadata."""
        chunk_text = "\n\n".join(el.text for el in elements if el.text.strip())
        
        # Get all unique page numbers
        page_numbers = sorted({el.page_num for el in elements})
        
        # Format page ranges (e.g., [1,2,3,5] -> "pages 1-3, 5")
        if len(page_numbers) > 1:
            ranges = []
            start = page_numbers[0]
            
            for i in range(1, len(page_numbers)):
                if page_numbers[i] != page_numbers[i-1] + 1:
                    if start == page_numbers[i-1]:
                        ranges.append(str(start))
                    else:
                        ranges.append(f"{start}-{page_numbers[i-1]}")
                    start = page_numbers[i]
            
            # Add the last range
            if start == page_numbers[-1]:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{page_numbers[-1]}")
            
            page_str = f"pages {' ,'.join(ranges)}"
        else:
            page_str = f"page {page_numbers[0]}"
        
        chunk_metadata = {
            'sources': page_str,
            'element_types': ', '.join(sorted({el.element_type for el in elements if el.element_type})),
            'is_continuation': False,
        }
        
        # Add metadata from the first element, ensuring all values are JSON-serializable
        if elements and elements[0].metadata:
            for key, value in elements[0].metadata.items():
                if isinstance(value, (list, tuple, set)):
                    # Convert lists/tuples/sets to comma-separated strings
                    chunk_metadata[key] = ', '.join(map(str, value))
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    # Keep primitives and None as-is
                    chunk_metadata[key] = value
                else:
                    # Convert any other type to string
                    chunk_metadata[key] = str(value)
            
        return {
            'page_content': chunk_text,
            'metadata': chunk_metadata
        }

class EnhancedPDFLoader:
    """Enhanced PDF loader with improved text extraction and parallel chunking."""
    
    def __init__(self, file_path: str, chunk_size: int = 512, chunk_overlap: int = 64, 
                 enable_ocr: bool = False, max_workers: int = None):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_workers = max_workers or os.cpu_count()
        self.processor = EnhancedPDFProcessor(
            enable_ocr=enable_ocr,
            max_workers=max(1, self.max_workers // 2)  # Reserve some workers for chunking
        )
        self.chunker = PDFChunkingStrategy(chunk_size, chunk_overlap)
        
    def load(self) -> List[Document]:
        """Load and process PDF document with parallel processing."""
        return list(self.lazy_load())
        
    def lazy_load(self) -> Generator[Document, None, None]:
        """Lazy load the document with parallel processing of pages."""
        logger.info(f"Processing PDF: {self.file_name}")
        start_time = time.time()
        
        try:
            # Process PDF with layout analysis
            elements = self.processor.process_pdf(self.file_path)
            
            if not elements:
                logger.warning(f"No content extracted from {self.file_name}")
                return
            
            # Group elements by page for parallel chunking
            pages = {}
            for el in elements:
                pages.setdefault(el.page_num, []).append(el)
            
            # Process pages in parallel
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                # Submit all pages for chunking
                future_to_page = {
                    executor.submit(self._chunk_page, page_num, page_elems): page_num
                    for page_num, page_elems in pages.items()
                }
                
                # Process chunks as they complete
                for future in concurrent.futures.as_completed(future_to_page):
                    try:
                        page_num = future_to_page[future]
                        chunks = future.result()
                        for chunk_idx, chunk in enumerate(chunks):
                            yield Document(
                                page_content=chunk['page_content'],
                                metadata={
                                    'source': self.file_name,
                                    'page': page_num,
                                    'chunk_id': chunk_idx,
                                    'document_type': 'pdf',
                                    'processing_time': datetime.utcnow().isoformat(),
                                    **chunk['metadata']
                                }
                            )
                    except Exception as e:
                        logger.error(f"Error processing page {page_num}: {e}")
            
            processing_time = time.time() - start_time
            logger.info(f"Processed {self.file_name} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing PDF {self.file_name}: {e}")
            raise
    
    def _chunk_page(self, page_num: int, elements: List[PDFElement]) -> List[Dict[str, Any]]:
        """Process a single page's elements into chunks (runs in a separate process)."""
        return self.chunker.chunk_elements(elements)

class DocumentLoader:
    """A unified document loader that handles different file types and directories with parallel processing."""
    
    def __init__(self, path: str, chunk_size: int = 512, chunk_overlap: int = 64, 
                 enable_ocr: bool = False, max_workers: int = None):
        """
        Initialize the document loader.
        
        Args:
            path: Path to a directory containing documents or a specific document file
            chunk_size: Target size for text chunks (in tokens)
            chunk_overlap: Overlap between chunks (in tokens)
            enable_ocr: Whether to enable OCR for image content in PDFs
            max_workers: Maximum number of worker processes to use for parallel processing
        """
        self.path = path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_ocr = enable_ocr
        self.max_workers = max_workers or os.cpu_count()

    def _process_single_file(self, file_path: str) -> List[Document]:
        """Process a single file and return its documents."""
        try:
            loader = EnhancedPDFLoader(
                file_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                enable_ocr=self.enable_ocr,
                max_workers=max(1, self.max_workers // 2)  # Reserve some workers for other files
            )
            return list(loader.lazy_load())
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []

    def load_documents(self) -> List[Document]:
        """
        Load and process documents from the specified path with parallel processing.
        
        Returns:
            List of processed documents with metadata
        """
        logger.info(f"Loading documents from: {self.path}")
        start_time = time.time()
        
        if os.path.isdir(self.path):
            # Get all PDF files in the directory
            pdf_files = []
            for root, _, files in os.walk(self.path):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))
            
            if not pdf_files:
                logger.warning(f"No PDF files found in {self.path}")
                return []
            
            logger.info(f"Found {len(pdf_files)} PDF files to process with {self.max_workers} workers")
            
            # Process files in parallel
            documents = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all files for processing
                future_to_file = {
                    executor.submit(self._process_single_file, file_path): file_path
                    for file_path in pdf_files
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_file):
                    try:
                        file_docs = future.result()
                        documents.extend(file_docs)
                        logger.info(f"Processed {os.path.basename(future_to_file[future])}: "
                                  f"{len(file_docs)} chunks")
                    except Exception as e:
                        file_path = future_to_file[future]
                        logger.error(f"Error processing file {file_path}: {e}")
        else:
            # Single file
            documents = self._process_single_file(self.path)
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(documents)} chunks in {processing_time:.2f} seconds")
        return documents

def load_documents(path: str, chunk_size: int = 512, chunk_overlap: int = 64, 
                  enable_ocr: bool = False, max_workers: int = None) -> List[Document]:
    """
    Load documents from the specified path with parallel processing.
    
    Args:
        path: Path to the directory containing documents or a specific document file
        chunk_size: Target size for text chunks (in tokens)
        chunk_overlap: Overlap between chunks (in tokens)
        enable_ocr: Whether to enable OCR for image content in PDFs
        max_workers: Maximum number of worker processes to use for parallel processing
        
    Returns:
        List of loaded documents with enhanced metadata
    """
    loader = DocumentLoader(
        path, 
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enable_ocr=enable_ocr,
        max_workers=max_workers
    )
    return loader.load_documents()
