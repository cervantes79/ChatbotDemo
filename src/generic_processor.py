"""
Generic Processor for LLM-free NLP processing
ICAR methodology enhancement
Author: Barış Genç
"""

import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessedChunk:
    """Represents a processed document chunk"""
    chunk_id: str
    chunk_type: str  # 'keywords', 'summary', 'hybrid'
    content: str
    original_text: str
    doc_id: str
    section: Optional[str]
    position: int
    confidence: float
    keywords: List[str]
    summary: str

class GenericProcessor:
    """
    Generic ICAR V2 processor with LLM-free NLP capabilities
    Uses NLTK and scikit-learn for cost-effective processing
    Author: Barış Genç
    """
    
    def __init__(self, processing_mode: str = "hybrid"):
        self.processing_mode = processing_mode  # 'keywords', 'summary', 'hybrid'
        self.chunk_size = 500
        self.overlap = 50
        self.min_chunk_size = 100
        
        # Initialize processing tools
        try:
            import nltk
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            from nltk.corpus import stopwords
            from nltk.tokenize import sent_tokenize, word_tokenize
            from nltk.stem import WordNetLemmatizer
            
            self.stopwords = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )
            
            self.nltk = nltk
            self.sklearn_available = True
            
        except ImportError as e:
            logger.error(f"Required NLP libraries not installed: {e}")
            self.sklearn_available = False
    
    def extract_keywords_tfidf(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords using TF-IDF"""
        if not self.sklearn_available:
            return self._fallback_keywords(text, top_k)
        
        try:
            # Clean and prepare text
            cleaned_text = self._clean_text(text)
            
            # Fit TF-IDF vectorizer
            tfidf_matrix = self.vectorizer.fit_transform([cleaned_text])
            
            # Get feature names and scores
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            word_scores = list(zip(feature_names, tfidf_scores))
            word_scores.sort(key=lambda x: x[1], reverse=True)
            
            keywords = [word for word, score in word_scores[:top_k] if score > 0]
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting TF-IDF keywords: {e}")
            return self._fallback_keywords(text, top_k)
    
    def _fallback_keywords(self, text: str, top_k: int) -> List[str]:
        """Fallback keyword extraction without scikit-learn"""
        try:
            # Simple frequency-based keyword extraction
            words = self._clean_text(text).split()
            word_freq = {}
            
            for word in words:
                if len(word) > 3 and word.lower() not in self.stopwords:
                    word_freq[word.lower()] = word_freq.get(word.lower(), 0) + 1
            
            # Sort by frequency
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in sorted_words[:top_k]]
            
        except Exception as e:
            logger.error(f"Error in fallback keyword extraction: {e}")
            return []
    
    def extractive_summarization(self, text: str, num_sentences: int = 3) -> str:
        """Extract summary sentences using scoring"""
        if not self.sklearn_available:
            return self._fallback_summary(text, num_sentences)
        
        try:
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Tokenize into sentences
            sentences = self.nltk.sent_tokenize(text)
            if len(sentences) <= num_sentences:
                return text
            
            # Create TF-IDF vectors for sentences
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence importance scores
            sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
            
            # Get top sentences
            top_indices = np.argsort(sentence_scores)[::-1][:num_sentences]
            top_indices.sort()  # Maintain original order
            
            summary_sentences = [sentences[i] for i in top_indices]
            return ' '.join(summary_sentences)
            
        except Exception as e:
            logger.error(f"Error in extractive summarization: {e}")
            return self._fallback_summary(text, num_sentences)
    
    def _fallback_summary(self, text: str, num_sentences: int) -> str:
        """Fallback summary without scikit-learn"""
        try:
            sentences = self.nltk.sent_tokenize(text)
            if len(sentences) <= num_sentences:
                return text
            
            # Simple heuristic: take first, middle, and last sentences
            step = max(1, len(sentences) // num_sentences)
            selected = []
            for i in range(0, len(sentences), step):
                if len(selected) < num_sentences:
                    selected.append(sentences[i])
            
            return ' '.join(selected)
            
        except Exception as e:
            logger.error(f"Error in fallback summary: {e}")
            return text[:500] + "..."
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        try:
            # Remove special characters and extra whitespace
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip().lower()
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text
    
    def chunk_document(self, text: str, doc_id: str) -> List[str]:
        """Split document into chunks"""
        try:
            if len(text) <= self.chunk_size:
                return [text]
            
            chunks = []
            start = 0
            
            while start < len(text):
                end = start + self.chunk_size
                
                if end >= len(text):
                    # Last chunk
                    chunk = text[start:]
                    if len(chunk.strip()) >= self.min_chunk_size:
                        chunks.append(chunk)
                    break
                
                # Try to break at sentence boundary
                chunk_text = text[start:end]
                sentences = self.nltk.sent_tokenize(chunk_text)
                
                if len(sentences) > 1:
                    # Use complete sentences
                    chunk = ' '.join(sentences[:-1])
                    chunks.append(chunk)
                    # Start next chunk with some overlap
                    start = start + len(chunk) - self.overlap
                else:
                    # No sentence boundary found, use the whole chunk
                    chunks.append(chunk_text)
                    start = end - self.overlap
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            # Fallback to simple chunking
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size-self.overlap)]
    
    def process_chunk(self, chunk_text: str, doc_id: str, position: int) -> ProcessedChunk:
        """Process a single chunk based on processing mode"""
        try:
            chunk_id = f"{doc_id[:8]}{position:04d}"
            
            # Extract keywords
            keywords = self.extract_keywords_tfidf(chunk_text, top_k=6)
            
            # Generate summary
            summary = self.extractive_summarization(chunk_text, num_sentences=2)
            
            # Create processed content based on mode
            if self.processing_mode == "keywords":
                content = f"{chunk_text[:100]}... | Keywords: {' '.join(keywords)}"
            elif self.processing_mode == "summary":
                content = summary
            else:  # hybrid
                content = f"{summary} | Keywords: {' '.join(keywords)}"
            
            return ProcessedChunk(
                chunk_id=chunk_id,
                chunk_type=self.processing_mode,
                content=content,
                original_text=chunk_text,
                doc_id=doc_id,
                section=None,
                position=position,
                confidence=1.0,
                keywords=keywords,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            # Return minimal chunk
            return ProcessedChunk(
                chunk_id=f"{doc_id[:8]}{position:04d}",
                chunk_type="basic",
                content=chunk_text,
                original_text=chunk_text,
                doc_id=doc_id,
                section=None,
                position=position,
                confidence=0.5,
                keywords=[],
                summary=chunk_text[:200] + "..."
            )
    
    def process_document(self, text: str, doc_id: str, metadata: Dict = None) -> List[ProcessedChunk]:
        """Process entire document into chunks"""
        try:
            chunks = self.chunk_document(text, doc_id)
            processed_chunks = []
            
            for i, chunk_text in enumerate(chunks):
                processed_chunk = self.process_chunk(chunk_text, doc_id, i)
                processed_chunks.append(processed_chunk)
            
            logger.info(f"Processed document {doc_id} into {len(processed_chunks)} chunks using {self.processing_mode} mode")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {e}")
            return []
    
    def reconstruct_context(self, chunks: List[ProcessedChunk], query: str) -> str:
        """Reconstruct original context from chunks"""
        try:
            if not chunks:
                return ""
            
            # Sort chunks by position
            sorted_chunks = sorted(chunks, key=lambda x: x.position)
            
            # Combine original text with enhanced metadata
            context_parts = []
            for chunk in sorted_chunks:
                context_part = f"[Document: {chunk.doc_id}, Position: {chunk.position}]\n"
                context_part += chunk.original_text
                if chunk.keywords:
                    context_part += f"\n[Key concepts: {', '.join(chunk.keywords[:3])}]"
                context_parts.append(context_part)
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error reconstructing context: {e}")
            return "\n\n".join([chunk.original_text for chunk in chunks])
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            "processing_mode": self.processing_mode,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "sklearn_available": self.sklearn_available,
            "nltk_available": hasattr(self, 'nltk'),
            "processor_type": "Generic ICAR V2 Processor by Barış Genç"
        }