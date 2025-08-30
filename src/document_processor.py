import os
import logging
from typing import List, Optional
try:
    from PyPDF2 import PdfReader
except ImportError:
    try:
        from pypdf import PdfReader
    except ImportError:
        PdfReader = None
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_pdf(self, file_path: str) -> Optional[str]:
        try:
            if PdfReader is None:
                logger.error("PyPDF2 not available, cannot process PDF files")
                return None
                
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text += page.extract_text() + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num} in {file_path}: {str(e)}")
                        continue
                
                if not text.strip():
                    logger.warning(f"No text extracted from {file_path}")
                    return None
                    
                return text
                
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            return None

    def load_text_file(self, file_path: str) -> Optional[str]:
        """Load text from a .txt file"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                
                if not text.strip():
                    logger.warning(f"No text found in {file_path}")
                    return None
                    
                return text
                
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            return None

    def process_documents(self, data_dir: str) -> List[Document]:
        documents = []
        
        try:
            if not os.path.exists(data_dir):
                logger.error(f"Data directory not found: {data_dir}")
                return documents
            
            # Get both PDF and text files
            all_files = os.listdir(data_dir)
            pdf_files = [f for f in all_files if f.endswith('.pdf')]
            txt_files = [f for f in all_files if f.endswith('.txt')]
            
            total_files = pdf_files + txt_files
            
            if not total_files:
                logger.warning(f"No PDF or text files found in {data_dir}")
                return documents
            
            logger.info(f"Found {len(pdf_files)} PDF files and {len(txt_files)} text files to process")
            
            # Process PDF files
            for filename in pdf_files:
                file_path = os.path.join(data_dir, filename)
                logger.info(f"Processing PDF: {filename}")
                
                text = self.load_pdf(file_path)
                if text:
                    chunks = self.text_splitter.split_text(text)
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": filename,
                                "file_type": "pdf",
                                "chunk_id": i,
                                "total_chunks": len(chunks)
                            }
                        )
                        documents.append(doc)
                        
                    logger.info(f"Created {len(chunks)} chunks from {filename}")
                else:
                    logger.error(f"Failed to process PDF {filename}")
            
            # Process text files
            for filename in txt_files:
                file_path = os.path.join(data_dir, filename)
                logger.info(f"Processing text file: {filename}")
                
                text = self.load_text_file(file_path)
                if text:
                    chunks = self.text_splitter.split_text(text)
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": filename,
                                "file_type": "txt",
                                "chunk_id": i,
                                "total_chunks": len(chunks)
                            }
                        )
                        documents.append(doc)
                        
                    logger.info(f"Created {len(chunks)} chunks from {filename}")
                else:
                    logger.error(f"Failed to process text file {filename}")
            
            logger.info(f"Total documents created: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return documents