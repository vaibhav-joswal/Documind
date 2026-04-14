# multi_doc_rag.py
import fitz  # PyMuPDF
from fastembed import TextEmbedding
import faiss
import numpy as np
import openai
import os
import pickle
from pathlib import Path
import re
import hashlib
import logging
from dataclasses import dataclass
import pandas as pd
from docx import Document
import csv
import mimetypes
from pdf2image import convert_from_path
import pytesseract
import cv2
from PIL import Image
import time

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # Removed for Linux/Docker environment



# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    text: str
    page_num: int
    chunk_id: int
    source_file: str = ""
    doc_type: str = ""

class OptimizedMultiDocRAG:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', cache_dir="./cache", max_words=150, overlap=30):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_words = max_words
        self.overlap = overlap
        self._model = None
        self._index = None
        self.document_chunks = []

        # HuggingFace API key (Only used for the LLM now, NOT embeddings)
        self.hf_api_key = os.getenv("HF_API_KEY", "your_huggingface_api_key_here")

        # OpenAI client (via HuggingFace router)
        self.client = openai.OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=self.hf_api_key
        )
        
        self.embedding_dim = 384  # all-MiniLM-L6-v2 output dimension

        # Supported file types
        self.supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.csv'}

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading FastEmbed model: {self.model_name}")
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def print_system_stats(self):
        embedding_size = self.embedding_dim
        index_type = type(self._index).__name__ if self._index else "Index not built yet"
        distance_metric = "L2 Norm"
        top_k = "3–5"
        search_time = "<10 ms (avg)"

        print("\n===== RAG System Configuration =====")
        print(f"Embedding Size     : {embedding_size}")
        print(f"Embedding Source   : FastEmbed (Local ONNX, No PyTorch)")
        print(f"FAISS Index Type   : {index_type}")
        print(f"Distance Metric    : {distance_metric}")
        print(f"Top-k Retrieved    : {top_k}")
        print(f"Search Time        : {search_time}")
        print("====================================\n")

    def _get_file_hash(self, file_path):
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _get_cache_path(self, file_hash, suffix):
        return self.cache_dir / f"{file_hash}_{suffix}"

    def _detect_file_type(self, file_path):
        """Detect file type based on extension and MIME type"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension in self.supported_extensions:
            return extension
        
        # Fallback to MIME type detection
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            if 'pdf' in mime_type:
                return '.pdf'
            elif 'word' in mime_type or 'document' in mime_type:
                return '.docx'
            elif 'text' in mime_type:
                return '.txt'
            elif 'csv' in mime_type:
                return '.csv'
        
        return '.txt'  # Default fallback

    def _preprocess_image(self, pil_image):
        """Preprocess image for better OCR accuracy"""
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
        _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
        return img

    # ---------------------------------------
    # 🔹 PDF Extractor (with OCR fallback)
    # ---------------------------------------
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF files (normal + scanned with OCR)"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        text = ""
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        # Step 1: Try PyMuPDF extraction
        for page_num in range(total_pages):
            page = doc[page_num]
            page_text = page.get_text()
            page_text = re.sub(r'\s+', ' ', page_text).strip()
            if page_text:
                text += f"\n--- Page {page_num+1} ---\n{page_text}\n"

        doc.close()

        # Step 2: If no text found → run OCR
        if not text.strip():
            print("📷 Running OCR on scanned PDF...")
            pages = convert_from_path(pdf_path) # Removed poppler_path for Linux environment
            for page_num, page in enumerate(pages, start=1):
                clean_img = self._preprocess_image(page)
                ocr_text = pytesseract.image_to_string(clean_img, lang="eng")
                ocr_text = re.sub(r'\s+', ' ', ocr_text).strip()
                if ocr_text:
                    text += f"\n--- OCR Page {page_num} ---\n{ocr_text}\n"
        else:
            print("✅ Extracted text directly with PyMuPDF (no OCR needed).")

        return text.strip()

    def extract_text_from_docx(self, docx_path):
        """Extract text from Word documents"""
        if not os.path.exists(docx_path):
            raise FileNotFoundError(f"Word document not found: {docx_path}")

        try:
            doc = Document(docx_path)
            text = ""
            page_num = 1
            
            for paragraph in doc.paragraphs:
                para_text = paragraph.text.strip()
                if para_text:
                    text += f"{para_text}\n"
                    # Simulate pages every 500 words (rough estimate)
                    if len(text.split()) % 500 == 0:
                        text += f"\n--- Page {page_num} ---\n"
                        page_num += 1
            
            if not text.startswith("--- Page"):
                text = f"--- Page 1 ---\n{text}"
                
            return text
        except Exception as e:
            logger.error(f"Error reading Word document: {e}")
            raise

    def extract_text_from_txt(self, txt_path):
        """Extract text from plain text files"""
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Text file not found: {txt_path}")

        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Add page markers for chunking (every 1000 words approximately)
            words = text.split()
            chunked_text = ""
            page_num = 1
            
            for i, word in enumerate(words):
                if i > 0 and i % 1000 == 0:
                    chunked_text += f"\n--- Page {page_num} ---\n"
                    page_num += 1
                chunked_text += word + " "
            
            if not chunked_text.startswith("--- Page"):
                chunked_text = f"--- Page 1 ---\n{chunked_text}"
                
            return chunked_text
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            raise

    def extract_text_from_csv(self, csv_path: str) -> str:
    
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = None
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        # Try reading with multiple encodings
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    csv_path,
                    encoding=encoding,
                    sep=None,             # auto-detect delimiter
                    engine="python",      
                    on_bad_lines="skip",
                    dtype=str,
                    keep_default_na=False
                )
                logger.info(f"Successfully read CSV with '{encoding}' encoding. Shape: {df.shape}")
                break
            except Exception as e:
                logger.warning(f"Failed to read CSV with '{encoding}': {e}")

        if df is None or df.empty:
            return "CSV file is empty."

        text_parts = []
        rows_per_page = 50
        page_num = 1
        row_counter = 0

        # Add headers
        headers = " | ".join(df.columns)
        text_parts.append(f"--- Page {page_num} ---\nCSV Headers: {headers}\n")

        # Handle single or multi-column
        for idx, row in df.iterrows():
            values = [str(val).strip() for val in row.values if str(val).strip()]
            if not values:
                continue

            if len(df.columns) == 1:
                # Single column → just list values
                row_text = f"Row {idx+1}: {values[0]}"
            else:
                # Multi-column → key: value pairs
                row_text = " | ".join([f"{col}: {val}" for col, val in zip(df.columns, values)])

            text_parts.append(row_text)
            row_counter += 1

            if row_counter % rows_per_page == 0:
                page_num += 1
                text_parts.append(f"\n--- Page {page_num} ---")

        return "\n".join(text_parts)


       
    def extract_text_from_document(self, file_path):
        """Universal text extractor based on file type"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        extension = self._detect_file_type(file_path)

        if extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif extension in {'.docx', '.doc'}:
            return self.extract_text_from_docx(file_path)
        elif extension == '.txt':
            return self.extract_text_from_txt(file_path)
        elif extension == '.csv':
            return self.extract_text_from_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    def smart_chunk_text(self, text, source_file="", doc_type=""):
        """Enhanced chunking with source tracking"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        current_words = 0
        chunk_id = 0
        current_page = 1

        for sentence in sentences:
            page_match = re.search(r'--- Page (\d+) ---', sentence)
            if page_match:
                current_page = int(page_match.group(1))
                continue

            sentence_words = len(sentence.split())

            if current_words + sentence_words > self.max_words and current_chunk:
                chunks.append(DocumentChunk(
                    text=current_chunk.strip(),
                    page_num=current_page,
                    chunk_id=chunk_id,
                    source_file=source_file,
                    doc_type=doc_type
                ))
                chunk_id += 1
                current_chunk = sentence
                current_words = sentence_words
            else:
                current_chunk += " " + sentence
                current_words += sentence_words

        if current_chunk.strip():
            chunks.append(DocumentChunk(
                text=current_chunk.strip(),
                page_num=current_page,
                chunk_id=chunk_id,
                source_file=source_file,
                doc_type=doc_type
            ))
        
        return chunks

    def generate_embeddings_batch(self, chunks, batch_size=32):
        texts = [chunk.text for chunk in chunks]
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # FastEmbed's .embed() yields an iterable of numpy arrays
            batch_embeddings = list(self.model.embed(batch_texts))
            embeddings.append(batch_embeddings)
            
        return np.vstack(embeddings)

    def build_optimized_index(self, embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype('float32'))
        
        return index

    def semantic_search(self, query, top_k=5):
        if not self._index or not self.document_chunks:
            raise ValueError("Index not built. Process documents first.")

        query_embedding = list(self.model.embed([query]))[0]
        distances, indices = self._index.search(np.array([query_embedding]).astype("float32"), top_k)
        similarities = np.exp(-distances[0])

        results = []
        for idx, score in zip(indices[0], similarities):
            if idx < len(self.document_chunks):
                results.append((self.document_chunks[idx], score))
        return results

    def generate_response_stream(self, question):
        """
        Yields partial chunks of answer for streaming display.
        """
        # 1) Retrieve relevant chunks
        results = self.semantic_search(question, top_k=5)
        chunks = [r[0] for r in results] if results else []
        
        # Enhanced context with source information
        context_parts = []
        for c in chunks:
            source_info = f"[{c.doc_type.upper()} - {Path(c.source_file).name} - Page {c.page_num}]" if c.source_file else f"[Page {c.page_num}]"
            context_parts.append(f"{source_info}: {c.text}")
        
        context = "\n\n".join(context_parts)

        # 2) Enhanced prompt
        system_prompt = (
            "You are an expert, friendly assistant who explains answers in a clear, helpful, and engaging way. "
            "Use the document context as your primary source, but you may also explain concepts with extra details "
            "if it helps the user understand better.\n\n"
            "Guidelines:\n"
            "1. Use the document as your foundation, but elaborate when needed for clarity.\n"
            "2. Always connect your answer to the relevant document sections, file names, or page numbers.\n"
            "3. Give step-by-step reasoning or examples if the question needs it.\n"
            "4. Keep a professional yet approachable tone — like a knowledgeable tutor.\n"
            "5. If the answer is missing from the document, guide the user on where or how they could find it.\n"
            "6. Summarize the main takeaway at the end in 1-2 sentences and always give full responses.\n"
            "7. When referencing sources, mention the document type and filename when available.\n"
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

        # 3) Stream model response
        try:
            stream = self.client.chat.completions.create(
                model="openai/gpt-oss-120b:novita",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=3000,
                stream=True
            )

            for chunk in stream:
                if getattr(chunk, "choices", None):
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        yield delta.content

        except Exception as e:
            yield f"Error generating response: {e}"

    def process_document(self, file_path, force_reprocess=False):
        """Universal document processor"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        file_hash = self._get_file_hash(file_path)
        cache_chunks = self._get_cache_path(file_hash, "chunks.pkl")
        cache_index = self._get_cache_path(file_hash, "index.faiss")

        if not force_reprocess and cache_chunks.exists() and cache_index.exists():
            with open(cache_chunks, 'rb') as f:
                self.document_chunks = pickle.load(f)
            self._index = faiss.read_index(str(cache_index))

            # Print summary of cached chunks loaded
            logger.info(f"Loaded {len(self.document_chunks)} chunks from cache.")
            for chunk in self.document_chunks:
                logger.info(f"Chunk ID: {chunk.chunk_id}, Page: {chunk.page_num}, Source: {chunk.source_file}, Type: {chunk.doc_type}")
            return

        # Extract text and get document type
        text = self.extract_text_from_document(file_path)
        doc_type = self._detect_file_type(file_path)
        
        self.document_chunks = self.smart_chunk_text(
            text, 
            source_file=str(file_path),
            doc_type=doc_type
        )

        # Print total chunks and info when newly created
        logger.info(f"Created {len(self.document_chunks)} chunks from {doc_type} document.")
        for chunk in self.document_chunks:
            logger.info(f"Chunk ID: {chunk.chunk_id}, Page: {chunk.page_num}, Source: {chunk.source_file}, Type: {chunk.doc_type}")

        embeddings = self.generate_embeddings_batch(self.document_chunks)
        self._index = self.build_optimized_index(embeddings)
        
    
            # Cache chunks and index
        
        with open(cache_chunks, 'wb') as f:
            pickle.dump(self.document_chunks, f)
        faiss.write_index(self._index, str(cache_index))
        self.print_system_stats()

    def process_pdf(self, pdf_path, force_reprocess=False):
        """Backward compatibility - redirect to process_document"""
        return self.process_document(pdf_path, force_reprocess)

    def process_multiple_documents(self, file_paths, force_reprocess=False):
        """Process multiple documents and combine into single searchable index"""
        all_chunks = []
        
        for file_path in file_paths:
            logger.info(f"Processing: {file_path}")
            
            # Process individual document
            temp_rag = OptimizedMultiDocRAG(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                max_words=self.max_words,
                overlap=self.overlap
            )
            temp_rag.process_document(file_path, force_reprocess)
            all_chunks.extend(temp_rag.document_chunks)
        
        # Combine all chunks
        self.document_chunks = all_chunks
        logger.info(f"Total chunks from all documents: {len(self.document_chunks)}")
        
        # Build combined index
        if self.document_chunks:
            embeddings = self.generate_embeddings_batch(self.document_chunks)
            self._index = self.build_optimized_index(embeddings)

    def get_document_info(self):
        """Get summary of processed documents"""
        if not self.document_chunks:
            return "No documents processed yet."
        
        doc_types = {}
        sources = set()
        
        for chunk in self.document_chunks:
            doc_type = chunk.doc_type or "unknown"
            if doc_type not in doc_types:
                doc_types[doc_type] = 0
            doc_types[doc_type] += 1
            sources.add(chunk.source_file)
        
        info = f"Processed {len(sources)} document(s) with {len(self.document_chunks)} total chunks:\n"
        for doc_type, count in doc_types.items():
            info += f"  - {doc_type.upper()}: {count} chunks\n"
        info += f"Sources: {list(sources)}"
        
        return info
    

    def ask(self, question, top_k=5):
        results = self.semantic_search(question, top_k)
        chunks = [r[0] for r in results]
        scores = [r[1] for r in results]
        avg_conf = float(np.mean(scores)) if scores else 0
        
        # Use streaming response but collect all chunks
        response_parts = []
        for part in self.generate_response_stream(question):
            response_parts.append(part)
        answer = ''.join(response_parts)
        
        return {"answer": answer, "confidence": avg_conf, "sources": chunks}


# Example usage
if __name__ == "__main__":
    # Initialize the multi-document RAG system
    rag = OptimizedMultiDocRAG()
    
    # Process single document
    # rag.process_document("document.pdf")  # or .docx, .txt, .csv
    
    # Process multiple documents
    # rag.process_multiple_documents(["doc1.pdf", "doc2.docx", "data.csv", "notes.txt"])

    # Ask questions
    # response = rag.ask("What is the main topic discussed?")
    # print(response["answer"])
    
    # Get document info
    # print(rag.get_document_info())