"""
Concept Extractor for ICAR methodology
Author: Barış Genç
"""

import json
import re
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

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
    Advanced concept extraction system for ICAR methodology
    Extracts meaningful concepts from documents and queries
    Author: Barış Genç
    """
    
    def __init__(self):
        self.concept_categories = {
            "business": ["policy", "procedure", "employee", "company", "work", "office", "meeting"],
            "technical": ["algorithm", "data", "structure", "programming", "software", "system"],
            "weather": ["weather", "temperature", "forecast", "rain", "sunny", "cloudy", "wind"],
            "greeting": ["hello", "hi", "hey", "good morning", "thanks", "bye", "goodbye"],
            "education": ["course", "syllabus", "assignment", "exam", "student", "grade", "class"],
            "healthcare": ["patient", "medical", "diagnosis", "treatment", "symptom", "medication"],
            "product": ["specification", "feature", "price", "warranty", "manual", "guide"],
            "weather_request": ["weather in", "temperature in", "forecast for", "what's the weather"]
        }
        
        self.concept_index = {}
        self.document_concepts = {}
    
    def extract_document_concepts(self, text: str, doc_id: str) -> List[Concept]:
        """Extract concepts from document text"""
        try:
            text_lower = text.lower()
            concepts = []
            
            # Extract concepts by category
            for category, keywords in self.concept_categories.items():
                category_weight = 0
                matched_keywords = []
                
                for keyword in keywords:
                    if keyword in text_lower:
                        weight = text_lower.count(keyword) / len(text.split()) * 10
                        category_weight += weight
                        matched_keywords.append(keyword)
                
                if category_weight > 0.1:
                    concept = Concept(
                        name=category,
                        weight=min(category_weight, 1.0),
                        category=category,
                        confidence=min(len(matched_keywords) / len(keywords), 1.0)
                    )
                    concepts.append(concept)
            
            # Store in document concepts
            self.document_concepts[doc_id] = concepts
            
            logger.info(f"Extracted {len(concepts)} concepts from document {doc_id}")
            return concepts
            
        except Exception as e:
            logger.error(f"Error extracting document concepts: {e}")
            return []
    
    def extract_query_concepts(self, query: str) -> List[Concept]:
        """Extract concepts from user query"""
        try:
            query_lower = query.lower().strip()
            concepts = []
            
            # Weather detection patterns
            weather_patterns = [
                r'weather.*in\s+([a-zA-Z\s]+)',
                r'what.*weather.*([a-zA-Z\s]+)',
                r'temperature.*in\s+([a-zA-Z\s]+)',
                r'forecast.*([a-zA-Z\s]+)'
            ]
            
            for pattern in weather_patterns:
                if re.search(pattern, query_lower):
                    concepts.append(Concept("weather_request", 0.9, "weather", 0.95))
                    break
            
            # Greeting detection
            greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "thanks", "thank you", "bye", "goodbye"]
            if any(greeting in query_lower for greeting in greetings) or len(query_lower) < 10:
                concepts.append(Concept("greeting", 0.8, "greeting", 0.9))
            
            # General concept extraction
            for category, keywords in self.concept_categories.items():
                if category in ["weather_request", "greeting"]:
                    continue
                    
                category_weight = 0
                matched_count = 0
                
                for keyword in keywords:
                    if keyword in query_lower:
                        category_weight += 0.2
                        matched_count += 1
                
                if category_weight > 0.1:
                    concept = Concept(
                        name=category,
                        weight=min(category_weight, 1.0),
                        category=category,
                        confidence=min(matched_count / len(keywords), 1.0)
                    )
                    concepts.append(concept)
            
            return concepts
            
        except Exception as e:
            logger.error(f"Error extracting query concepts: {e}")
            return []
    
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