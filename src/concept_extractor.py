"""
Concept Extractor for ICAR methodology - Academic Paper Implementation
NLTK + TF-IDF + Named Entity Recognition
Author: Barış Genç
"""

import json
import re
import nltk
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('chunkers/maxent_ne_chunker_tab')
except LookupError:
    nltk.download('maxent_ne_chunker_tab')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

@dataclass
class Concept:
    """Represents an extracted concept with metadata"""
    name: str
    weight: float
    category: str = "general"
    confidence: float = 1.0
    
class ConceptExtractor:
    """
    Academic Paper Implementation - ICAR Concept Extraction
    NLTK + TF-IDF + Named Entity Recognition
    Author: Barış Genç
    """

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')

        self.concept_categories = {
            "business": [
                "policy", "procedure", "employee", "company", "work", "office", "meeting",
                "salary", "benefits", "vacation", "hours", "schedule", "department",
                "manager", "team", "project", "deadline", "budget", "revenue"
            ],
            "technical": [
                "algorithm", "data", "structure", "programming", "software", "system",
                "database", "server", "network", "security", "bug", "feature",
                "deployment", "testing", "debugging", "optimization", "performance"
            ],
            "weather": [
                "weather", "temperature", "forecast", "rain", "sunny", "cloudy", "wind",
                "storm", "snow", "humidity", "pressure", "climate", "hot", "cold"
            ],
            "greeting": [
                "hello", "hi", "hey", "good morning", "good afternoon", "thanks",
                "thank you", "bye", "goodbye", "please", "excuse me"
            ],
            "education": [
                "course", "syllabus", "assignment", "exam", "student", "grade", "class",
                "homework", "lecture", "professor", "university", "degree", "study"
            ],
            "healthcare": [
                "patient", "medical", "diagnosis", "treatment", "symptom", "medication",
                "doctor", "hospital", "clinic", "health", "disease", "therapy"
            ],
            "product": [
                "specification", "feature", "price", "warranty", "manual", "guide",
                "quality", "brand", "model", "version", "catalog", "description"
            ]
        }

        # Weather-specific patterns for NLTK processing
        self.weather_patterns = [
            r"weather in (\w+)",
            r"temperature in (\w+)",
            r"forecast for (\w+)",
            r"what's the weather",
            r"how's the weather",
            r"weather like in"
        ]

        self.concept_index = {}
        self.document_concepts = {}
    
    def extract_document_concepts(self, text: str, doc_id: str) -> List[Concept]:
        """Extract concepts from document text using NLTK + TF-IDF + NER"""
        try:
            text_lower = text.lower()
            concepts = []

            # 1. NLTK Named Entity Recognition
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            named_entities = ne_chunk(pos_tags)

            # Extract named entities
            for chunk in named_entities:
                if hasattr(chunk, 'label'):
                    entity_name = ' '.join([token for token, pos in chunk.leaves()]).lower()
                    concepts.append(Concept(
                        name=entity_name,
                        weight=2.0,  # NER entities get high weight
                        category="entity",
                        confidence=0.9
                    ))

            # 2. TF-IDF Important Terms
            try:
                # Fit and transform text for TF-IDF
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                scores = tfidf_matrix.toarray()[0]

                # Get top TF-IDF terms
                top_indices = scores.argsort()[-10:][::-1]  # Top 10 terms
                for idx in top_indices:
                    if scores[idx] > 0:
                        term = feature_names[idx]
                        concepts.append(Concept(
                            name=term,
                            weight=scores[idx] * 3.0,  # Scale TF-IDF score
                            category="tfidf",
                            confidence=0.8
                        ))
            except:
                pass  # Skip TF-IDF if not enough text

            # 3. Domain-specific keyword matching with word boundaries
            for category, keywords in self.concept_categories.items():
                for keyword in keywords:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                        weight = text_lower.count(keyword) * self._get_keyword_importance(keyword)
                        confidence = self._calculate_confidence(keyword, text_lower)

                        concepts.append(Concept(
                            name=keyword,
                            weight=weight,
                            category=category,
                            confidence=confidence
                        ))

            # Remove duplicates and sort by weight
            unique_concepts = {}
            for concept in concepts:
                if concept.name not in unique_concepts or concept.weight > unique_concepts[concept.name].weight:
                    unique_concepts[concept.name] = concept

            final_concepts = sorted(unique_concepts.values(), key=lambda x: x.weight, reverse=True)

            # Store in document concepts
            self.document_concepts[doc_id] = final_concepts

            logger.info(f"Extracted {len(final_concepts)} concepts from document {doc_id}")
            return final_concepts

        except Exception as e:
            logger.error(f"Error extracting document concepts: {e}")
            return []
    
    def extract_query_concepts(self, query: str) -> List[Concept]:
        """Extract concepts from user query using NLTK + TF-IDF + NER"""
        try:
            query_lower = query.lower().strip()
            concepts = []

            # 1. NLTK Named Entity Recognition
            tokens = word_tokenize(query)
            pos_tags = pos_tag(tokens)
            named_entities = ne_chunk(pos_tags)

            # Extract named entities
            for chunk in named_entities:
                if hasattr(chunk, 'label'):
                    entity_name = ' '.join([token for token, pos in chunk.leaves()]).lower()
                    concepts.append(Concept(
                        name=entity_name,
                        weight=2.0,  # NER entities get high weight
                        category="entity",
                        confidence=0.9
                    ))

            # 2. TF-IDF Important Terms
            try:
                # Fit and transform query for TF-IDF
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([query])
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                scores = tfidf_matrix.toarray()[0]

                # Get top TF-IDF terms
                top_indices = scores.argsort()[-5:][::-1]  # Top 5 terms
                for idx in top_indices:
                    if scores[idx] > 0:
                        term = feature_names[idx]
                        concepts.append(Concept(
                            name=term,
                            weight=scores[idx] * 3.0,  # Scale TF-IDF score
                            category="tfidf",
                            confidence=0.8
                        ))
            except:
                pass  # Skip TF-IDF if not enough text

            # 3. Weather patterns (special case)
            for pattern in self.weather_patterns:
                if re.search(pattern, query_lower):
                    concepts.append(Concept(
                        name="weather_query",
                        weight=3.0,
                        category="weather",
                        confidence=0.9
                    ))

            # 4. Domain-specific keyword matching with word boundaries
            for category, keywords in self.concept_categories.items():
                for keyword in keywords:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                        weight = query_lower.count(keyword) * self._get_keyword_importance(keyword)
                        confidence = self._calculate_confidence(keyword, query_lower)

                        concepts.append(Concept(
                            name=keyword,
                            weight=weight,
                            category=category,
                            confidence=confidence
                        ))

            # Remove duplicates and sort by weight
            unique_concepts = {}
            for concept in concepts:
                if concept.name not in unique_concepts or concept.weight > unique_concepts[concept.name].weight:
                    unique_concepts[concept.name] = concept

            return sorted(unique_concepts.values(), key=lambda x: x.weight, reverse=True)

        except Exception as e:
            logger.error(f"Error extracting query concepts: {e}")
            return []

    def _get_keyword_importance(self, keyword: str) -> float:
        """Assign importance weights to different keywords"""
        high_importance = ["weather", "temperature", "policy", "employee", "system", "algorithm", "security"]
        medium_importance = ["work", "data", "meeting", "project", "feature", "performance"]

        if keyword in high_importance:
            return 2.0
        elif keyword in medium_importance:
            return 1.5
        else:
            return 1.0

    def _calculate_confidence(self, keyword: str, text: str) -> float:
        """Calculate confidence based on context"""
        # Higher confidence if keyword appears with related terms
        context_words = text.split()
        keyword_index = -1

        for i, word in enumerate(context_words):
            if keyword in word:
                keyword_index = i
                break

        if keyword_index == -1:
            return 0.5

        # Check surrounding words for context
        context_boost = 0.0
        start = max(0, keyword_index - 2)
        end = min(len(context_words), keyword_index + 3)

        for category, keywords in self.concept_categories.items():
            for kw in keywords:
                if any(kw in context_words[i] for i in range(start, end) if i != keyword_index):
                    context_boost += 0.1

        return min(1.0, 0.6 + context_boost)
    
    def match_concepts(self, query_concepts: List[Concept], similarity_threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Match query concepts with document concepts"""
        try:
            matches = []
            
            for query_concept in query_concepts:
                for doc_id, doc_concepts in self.document_concepts.items():
                    for doc_concept in doc_concepts:
                        # Exact match
                        if query_concept.name == doc_concept.name:
                            score = query_concept.weight * doc_concept.weight * 1.0
                            matches.append((doc_id, score))
                        
                        # Category match
                        elif query_concept.category == doc_concept.category:
                            score = query_concept.weight * doc_concept.weight * 0.7
                            matches.append((doc_id, score))
            
            # Sort by score and filter by threshold
            matches = [(doc_id, score) for doc_id, score in matches if score >= similarity_threshold]
            matches.sort(key=lambda x: x[1], reverse=True)
            
            return matches[:5]  # Return top 5 matches
            
        except Exception as e:
            logger.error(f"Error matching concepts: {e}")
            return []
    
    def save_concept_index(self, filepath: str):
        """Save concept index to file"""
        try:
            index_data = {
                "concept_categories": self.concept_categories,
                "document_concepts": {
                    doc_id: [
                        {
                            "name": concept.name,
                            "weight": concept.weight,
                            "category": concept.category,
                            "confidence": concept.confidence
                        }
                        for concept in concepts
                    ]
                    for doc_id, concepts in self.document_concepts.items()
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Concept index saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving concept index: {e}")
    
    def load_concept_index(self, filepath: str):
        """Load concept index from file"""
        try:
            if not Path(filepath).exists():
                logger.info("No existing concept index found, starting fresh")
                return
            
            with open(filepath, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            if "concept_categories" in index_data:
                self.concept_categories.update(index_data["concept_categories"])
            
            if "document_concepts" in index_data:
                self.document_concepts = {}
                for doc_id, concept_data in index_data["document_concepts"].items():
                    concepts = []
                    for data in concept_data:
                        concept = Concept(
                            name=data["name"],
                            weight=data["weight"],
                            category=data["category"],
                            confidence=data["confidence"]
                        )
                        concepts.append(concept)
                    self.document_concepts[doc_id] = concepts
            
            logger.info(f"Concept index loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading concept index: {e}")
    
    def get_stats(self) -> Dict:
        """Get concept extraction statistics"""
        total_concepts = sum(len(concepts) for concepts in self.document_concepts.values())
        
        return {
            "total_documents": len(self.document_concepts),
            "total_concepts": total_concepts,
            "concept_categories": len(self.concept_categories),
            "average_concepts_per_doc": total_concepts / max(len(self.document_concepts), 1)
        }