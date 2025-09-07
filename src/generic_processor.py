import logging
import re
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
import json
import os
from pathlib import Path
import hashlib

# NLP libraries - no LLM dependencies
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.stem import WordNetLemmatizer
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('wordnet', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ProcessedChunk:
    """Generic processed chunk - can be keyword-based or summary-based"""
    chunk_id: str
    chunk_type: str  # "keywords", "summary", "hybrid"
    content: str  # processed content (keywords or summary)
    original_text: str  # original document section
    doc_id: str
    section: Optional[str] = None
    position: int = 0
    confidence: float = 1.0
    keywords: List[str] = None
    summary: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "content": self.content,
            "original_text": self.original_text,
            "doc_id": self.doc_id,
            "section": self.section,
            "position": self.position,
            "confidence": self.confidence,
            "keywords": self.keywords or [],
            "summary": self.summary or ""
        }

class GenericNLPProcessor:
    """
    Generic ICAR V2 - Domain Agnostic NLP Processor
    LLM-Free, Language-Independent Document Processing
    
    Author: Barış Genç
    """
    
    def __init__(self, processing_mode: str = "hybrid"):
        """
        Initialize Generic NLP Processor
        
        Args:
            processing_mode: "keywords", "summary", or "hybrid"
        """
        self.processing_mode = processing_mode
        self.lemmatizer = None
        self.stop_words = set()
        self.tfidf_vectorizer = None
        self.language = "english"  # Default, can be auto-detected
        
        # Initialize NLP components
        self._initialize_nlp()
        
        logger.info(f"Generic ICAR V2 Processor initialized - Mode: {processing_mode}")
    
    def _initialize_nlp(self):
        """Initialize NLP components without LLM dependencies"""
        if NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words(self.language))
                logger.info("NLTK components loaded successfully")
            except Exception as e:
                logger.warning(f"NLTK initialization failed: {e}")
                self._fallback_initialization()
        else:
            logger.warning("NLTK not available, using fallback methods")
            self._fallback_initialization()
        
        # Initialize TF-IDF for generic text processing
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),  # unigrams and bigrams
            lowercase=True,
            strip_accents='ascii'
        )
    
    def _fallback_initialization(self):
        """Fallback initialization when NLTK is not available"""
        # Basic English stop words
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
            'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above',
            'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
    
    def process_document(self, text: str, doc_id: str, source: str = "") -> List[ProcessedChunk]:
        """
        Process a document into chunks using generic NLP methods
        
        Args:
            text: Document text
            doc_id: Unique document identifier
            source: Source file/location
            
        Returns:
            List of ProcessedChunk objects
        """
        try:
            chunks = []
            
            # Split document into logical sections
            sections = self._split_into_sections(text)
            
            for i, section in enumerate(sections):
                if len(section.strip()) < 50:  # Skip very short sections
                    continue
                
                chunk_id = self._generate_chunk_id(doc_id, i)
                
                if self.processing_mode == "keywords":
                    chunk = self._create_keyword_chunk(section, chunk_id, doc_id, i)
                elif self.processing_mode == "summary":
                    chunk = self._create_summary_chunk(section, chunk_id, doc_id, i)
                else:  # hybrid
                    chunk = self._create_hybrid_chunk(section, chunk_id, doc_id, i)
                
                if chunk:
                    chunks.append(chunk)
            
            logger.info(f"Processed document {doc_id} into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {str(e)}")
            return []
    
    def _split_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections for processing"""
        # Multiple splitting strategies
        
        # Strategy 1: Split by paragraphs
        paragraphs = text.split('\n\n')
        
        # Strategy 2: Split by sentences if paragraphs are too long
        sections = []
        for para in paragraphs:
            if len(para) > 1000:  # Long paragraph
                if NLTK_AVAILABLE:
                    sentences = sent_tokenize(para)
                else:
                    sentences = re.split(r'[.!?]+', para)
                
                # Group sentences into chunks of ~500 characters
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk + sent) < 800:
                        current_chunk += sent + " "
                    else:
                        if current_chunk.strip():
                            sections.append(current_chunk.strip())
                        current_chunk = sent + " "
                
                if current_chunk.strip():
                    sections.append(current_chunk.strip())
            else:
                sections.append(para.strip())
        
        # Filter out very short sections
        return [s for s in sections if len(s) > 50]
    
    def _create_keyword_chunk(self, text: str, chunk_id: str, doc_id: str, position: int) -> ProcessedChunk:
        """Create keyword-based chunk"""
        try:
            keywords = self._extract_keywords(text)
            keyword_content = " ".join(keywords)
            
            return ProcessedChunk(
                chunk_id=chunk_id,
                chunk_type="keywords",
                content=keyword_content,
                original_text=text,
                doc_id=doc_id,
                position=position,
                keywords=keywords,
                confidence=self._calculate_confidence(keywords, text)
            )
            
        except Exception as e:
            logger.error(f"Error creating keyword chunk: {str(e)}")
            return None
    
    def _create_summary_chunk(self, text: str, chunk_id: str, doc_id: str, position: int) -> ProcessedChunk:
        """Create summary-based chunk using extractive summarization"""
        try:
            summary = self._extractive_summarization(text)
            
            return ProcessedChunk(
                chunk_id=chunk_id,
                chunk_type="summary",
                content=summary,
                original_text=text,
                doc_id=doc_id,
                position=position,
                summary=summary,
                confidence=self._calculate_confidence([summary], text)
            )
            
        except Exception as e:
            logger.error(f"Error creating summary chunk: {str(e)}")
            return None
    
    def _create_hybrid_chunk(self, text: str, chunk_id: str, doc_id: str, position: int) -> ProcessedChunk:
        """Create hybrid chunk with both keywords and summary"""
        try:
            keywords = self._extract_keywords(text)
            summary = self._extractive_summarization(text)
            
            # Combine keywords and summary for content
            hybrid_content = f"{summary} | Keywords: {' '.join(keywords)}"
            
            return ProcessedChunk(
                chunk_id=chunk_id,
                chunk_type="hybrid",
                content=hybrid_content,
                original_text=text,
                doc_id=doc_id,
                position=position,
                keywords=keywords,
                summary=summary,
                confidence=self._calculate_confidence(keywords + [summary], text)
            )
            
        except Exception as e:
            logger.error(f"Error creating hybrid chunk: {str(e)}")
            return None
    
    def _extract_keywords(self, text: str, max_keywords: int = 15) -> List[str]:
        """Extract keywords using TF-IDF and linguistic analysis"""
        try:
            keywords = []
            
            # Method 1: TF-IDF based extraction
            tfidf_keywords = self._tfidf_keywords(text)
            keywords.extend(tfidf_keywords[:max_keywords//2])
            
            # Method 2: Linguistic pattern based extraction
            linguistic_keywords = self._linguistic_keywords(text)
            keywords.extend(linguistic_keywords[:max_keywords//2])
            
            # Remove duplicates and clean
            keywords = list(dict.fromkeys(keywords))  # Preserve order, remove dupes
            keywords = [kw for kw in keywords if len(kw) > 2 and kw.lower() not in self.stop_words]
            
            return keywords[:max_keywords]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return self._fallback_keywords(text)
    
    def _tfidf_keywords(self, text: str) -> List[str]:
        """Extract keywords using TF-IDF"""
        try:
            # Fit TF-IDF on the single document
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            keyword_scores = list(zip(feature_names, tfidf_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [kw for kw, score in keyword_scores[:10] if score > 0]
            
        except Exception:
            return []
    
    def _linguistic_keywords(self, text: str) -> List[str]:
        """Extract keywords using linguistic patterns"""
        try:
            keywords = []
            
            if NLTK_AVAILABLE and self.lemmatizer:
                # Tokenize and POS tag
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)
                
                # Extract nouns, proper nouns, and adjectives
                for word, pos in pos_tags:
                    if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']:
                        word_clean = self.lemmatizer.lemmatize(word.lower())
                        if len(word_clean) > 2 and word_clean.isalpha():
                            keywords.append(word_clean)
            else:
                # Fallback: simple pattern matching
                # Find capitalized words (proper nouns)
                proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
                keywords.extend(proper_nouns)
                
                # Find repeated words (important terms)
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                word_freq = Counter(words)
                frequent_words = [word for word, freq in word_freq.most_common(10) 
                                if word not in self.stop_words]
                keywords.extend(frequent_words)
            
            return keywords
            
        except Exception:
            return []
    
    def _extractive_summarization(self, text: str, num_sentences: int = 2) -> str:
        """Create extractive summary by selecting key sentences"""
        try:
            if NLTK_AVAILABLE:
                sentences = sent_tokenize(text)
            else:
                sentences = re.split(r'[.!?]+', text)
            
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if len(sentences) <= num_sentences:
                return " ".join(sentences)
            
            # Score sentences based on keyword frequency
            keywords = self._extract_keywords(text, max_keywords=20)
            keyword_set = set(kw.lower() for kw in keywords)
            
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                score = 0
                
                # Score based on keyword presence
                for keyword in keyword_set:
                    if keyword in sentence_lower:
                        score += 1
                
                # Bonus for sentence position (first and last sentences often important)
                if i == 0:
                    score += 0.5
                elif i == len(sentences) - 1:
                    score += 0.3
                
                sentence_scores.append((sentence, score))
            
            # Select top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            selected_sentences = [s for s, score in sentence_scores[:num_sentences]]
            
            # Maintain original order
            summary_sentences = []
            for sentence in sentences:
                if sentence in selected_sentences:
                    summary_sentences.append(sentence)
            
            return " ".join(summary_sentences)
            
        except Exception as e:
            logger.error(f"Error in extractive summarization: {str(e)}")
            return text[:200] + "..."  # Fallback to truncation
    
    def _fallback_keywords(self, text: str) -> List[str]:
        """Fallback keyword extraction using simple methods"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = Counter(words)
        
        # Remove stop words and get frequent words
        keywords = []
        for word, freq in word_freq.most_common(15):
            if word not in self.stop_words and len(word) > 2:
                keywords.append(word)
        
        return keywords
    
    def _calculate_confidence(self, extracted_items: List[str], original_text: str) -> float:
        """Calculate confidence score for extraction quality"""
        try:
            if not extracted_items:
                return 0.0
            
            # Simple confidence based on coverage
            total_chars = sum(len(item) for item in extracted_items)
            text_length = len(original_text)
            
            if text_length == 0:
                return 0.0
            
            # Confidence = proportion of text covered, capped at 1.0
            confidence = min(1.0, total_chars / text_length * 10)  # Scale factor
            return round(confidence, 2)
            
        except Exception:
            return 0.5  # Default confidence
    
    def _generate_chunk_id(self, doc_id: str, position: int) -> str:
        """Generate unique chunk ID"""
        content = f"{doc_id}_{position}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def process_query(self, query: str) -> ProcessedChunk:
        """Process user query using same methodology as documents"""
        try:
            chunk_id = self._generate_chunk_id("query", 0)
            
            if self.processing_mode == "keywords":
                return self._create_keyword_chunk(query, chunk_id, "query", 0)
            elif self.processing_mode == "summary":
                return self._create_summary_chunk(query, chunk_id, "query", 0)
            else:  # hybrid
                return self._create_hybrid_chunk(query, chunk_id, "query", 0)
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return None
    
    def calculate_similarity(self, query_chunk: ProcessedChunk, doc_chunks: List[ProcessedChunk]) -> List[Tuple[ProcessedChunk, float]]:
        """Calculate similarity between query and document chunks"""
        try:
            if not query_chunk or not doc_chunks:
                return []
            
            # Prepare texts for similarity calculation
            all_texts = [query_chunk.content] + [chunk.content for chunk in doc_chunks]
            
            # Calculate TF-IDF similarity
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Create results with similarity scores
            results = []
            for i, chunk in enumerate(doc_chunks):
                similarity_score = similarity_matrix[i]
                
                # Boost score based on chunk confidence
                boosted_score = similarity_score * chunk.confidence
                
                results.append((chunk, boosted_score))
            
            # Sort by similarity score
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            "processor_version": "Generic ICAR V2",
            "processing_mode": self.processing_mode,
            "nltk_available": NLTK_AVAILABLE,
            "language": self.language,
            "author": "Barış Genç"
        }