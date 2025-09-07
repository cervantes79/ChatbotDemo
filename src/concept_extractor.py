import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

@dataclass
class Concept:
    """Represents a key concept extracted from text"""
    name: str
    weight: float
    context: str
    category: str = "general"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight,
            "context": self.context,
            "category": self.category
        }

@dataclass
class ConceptMatch:
    """Represents a match between query concepts and document concepts"""
    concept_name: str
    query_weight: float
    doc_weight: float
    similarity_score: float
    match_type: str  # exact, semantic, related
    
class ConceptExtractor:
    """
    Intelligent Concept-Aware RAG (ICAR) System
    Extracts and matches concepts for enhanced document retrieval
    
    Author: Barış Genç
    """
    
    def __init__(self):
        # Initialize LLM only if API key is available
        try:
            self.llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
        except Exception:
            self.llm = None
            logger.warning("OpenAI API key not found, concept extraction will use fallback methods")
        
        self.concept_index = defaultdict(dict)
        
        # Predefined concept categories for better organization
        self.concept_categories = {
            "employee_policies": ["work_hours", "remote_work", "vacation", "sick_leave", "benefits"],
            "company_info": ["contact", "departments", "leadership", "mission", "values"],
            "products": ["features", "specifications", "pricing", "support", "warranty"],
            "procedures": ["how_to", "guidelines", "requirements", "process", "steps"]
        }
    
    def extract_document_concepts(self, text: str, source: str = "") -> List[Concept]:
        """Extract key concepts from document text using ICAR methodology"""
        try:
            if self.llm is None:
                # Fallback to rule-based extraction
                logger.info("Using fallback concept extraction (no LLM available)")
                return self._fallback_concept_extraction(text)
            
            # Use LLM to extract structured concepts
            system_prompt = """You are an expert at extracting key concepts from business documents. 
Extract the most important concepts from the given text and categorize them.

For each concept, provide:
1. Concept name (2-4 words, snake_case)
2. Weight (0.0-1.0 based on importance)
3. Context (brief description of what it covers)
4. Category (employee_policies, company_info, products, procedures, or general)

Return ONLY a JSON array of concepts, no other text:"""

            user_prompt = f"""Text to analyze:
{text[:2000]}...

Extract 5-8 key concepts from this text."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            
            # Parse LLM response
            try:
                concepts_data = json.loads(response.content)
                concepts = []
                
                for concept_data in concepts_data:
                    concept = Concept(
                        name=concept_data.get("name", "unknown_concept"),
                        weight=float(concept_data.get("weight", 0.5)),
                        context=concept_data.get("context", ""),
                        category=concept_data.get("category", "general")
                    )
                    concepts.append(concept)
                
                # Enhance with statistical analysis
                concepts = self._enhance_with_tfidf(text, concepts)
                
                logger.info(f"Extracted {len(concepts)} concepts from document {source}")
                return concepts
                
            except json.JSONDecodeError:
                # Fallback to rule-based extraction
                logger.warning("LLM response parsing failed, using fallback extraction")
                return self._fallback_concept_extraction(text)
                
        except Exception as e:
            logger.error(f"Error extracting document concepts: {str(e)}")
            return self._fallback_concept_extraction(text)
    
    def extract_query_concepts(self, query: str) -> List[Concept]:
        """Extract concepts from user query with intent recognition"""
        try:
            query_lower = query.lower().strip()
            concepts = []
            
            # Rule-based concept extraction for common patterns
            concept_patterns = {
                "employee_benefits": r"(benefit|insurance|401k|health|dental|vision|gym)",
                "work_schedule": r"(work hours|schedule|time|office hours|remote)",
                "vacation_policy": r"(vacation|holiday|time off|leave|pto)",
                "contact_info": r"(contact|phone|email|address|support|help)",
                "product_features": r"(feature|specification|capability|function|what does)",
                "procedures": r"(how to|process|procedure|step|guideline|instruction)",
                "weather_request": r"(weather|temperature|forecast|rain|sunny|cloudy)",
                "greeting": r"(hello|hi|hey|good morning|good afternoon)"
            }
            
            for concept_name, pattern in concept_patterns.items():
                if re.search(pattern, query_lower):
                    weight = 0.8 if len(re.findall(pattern, query_lower)) > 1 else 0.6
                    concepts.append(Concept(
                        name=concept_name,
                        weight=weight,
                        context=f"Query contains keywords: {pattern}",
                        category=self._get_concept_category(concept_name)
                    ))
            
            # If no specific patterns found, use general intent analysis
            if not concepts:
                if len(query_lower) < 20:
                    concepts.append(Concept("general_query", 0.5, "Short general question", "general"))
                else:
                    concepts.append(Concept("specific_inquiry", 0.7, "Detailed question", "general"))
            
            logger.info(f"Extracted {len(concepts)} concepts from query")
            return concepts
            
        except Exception as e:
            logger.error(f"Error extracting query concepts: {str(e)}")
            return [Concept("unknown_intent", 0.5, "Failed to parse query", "general")]
    
    def match_concepts(self, query_concepts: List[Concept], document_concepts: List[Concept]) -> List[ConceptMatch]:
        """Match query concepts with document concepts using multiple strategies"""
        matches = []
        
        try:
            for query_concept in query_concepts:
                for doc_concept in document_concepts:
                    # Exact match
                    if query_concept.name == doc_concept.name:
                        matches.append(ConceptMatch(
                            concept_name=query_concept.name,
                            query_weight=query_concept.weight,
                            doc_weight=doc_concept.weight,
                            similarity_score=1.0,
                            match_type="exact"
                        ))
                    
                    # Category match
                    elif query_concept.category == doc_concept.category and query_concept.category != "general":
                        similarity = self._calculate_concept_similarity(query_concept.name, doc_concept.name)
                        if similarity > 0.3:
                            matches.append(ConceptMatch(
                                concept_name=f"{query_concept.name}->{doc_concept.name}",
                                query_weight=query_concept.weight,
                                doc_weight=doc_concept.weight,
                                similarity_score=similarity * 0.7,  # Category matches get lower score
                                match_type="semantic"
                            ))
            
            # Sort matches by combined score
            matches.sort(key=lambda x: x.similarity_score * (x.query_weight + x.doc_weight), reverse=True)
            logger.info(f"Found {len(matches)} concept matches")
            return matches
            
        except Exception as e:
            logger.error(f"Error matching concepts: {str(e)}")
            return []
    
    def build_concept_index(self, documents: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive concept index from all documents"""
        concept_index = defaultdict(lambda: {
            "documents": [],
            "total_weight": 0.0,
            "related_concepts": set(),
            "category": "general"
        })
        
        try:
            for doc in documents:
                doc_id = doc.get("id", "unknown")
                concepts = doc.get("concepts", [])
                
                for concept in concepts:
                    concept_name = concept.name if hasattr(concept, 'name') else concept.get("name")
                    concept_weight = concept.weight if hasattr(concept, 'weight') else concept.get("weight", 0.5)
                    concept_category = concept.category if hasattr(concept, 'category') else concept.get("category", "general")
                    
                    concept_index[concept_name]["documents"].append({
                        "doc_id": doc_id,
                        "weight": concept_weight,
                        "context": concept.context if hasattr(concept, 'context') else concept.get("context", "")
                    })
                    concept_index[concept_name]["total_weight"] += concept_weight
                    concept_index[concept_name]["category"] = concept_category
                    
                    # Build relationships
                    for other_concept in concepts:
                        other_name = other_concept.name if hasattr(other_concept, 'name') else other_concept.get("name")
                        if other_name != concept_name:
                            concept_index[concept_name]["related_concepts"].add(other_name)
            
            # Convert sets to lists for JSON serialization
            for concept_name in concept_index:
                concept_index[concept_name]["related_concepts"] = list(concept_index[concept_name]["related_concepts"])
            
            logger.info(f"Built concept index with {len(concept_index)} concepts")
            self.concept_index = concept_index
            return dict(concept_index)
            
        except Exception as e:
            logger.error(f"Error building concept index: {str(e)}")
            return {}
    
    def get_concept_score(self, query_concepts: List[Concept], document_concepts: List[Concept]) -> float:
        """Calculate overall concept-based relevance score for a document"""
        if not query_concepts or not document_concepts:
            return 0.0
        
        matches = self.match_concepts(query_concepts, document_concepts)
        
        if not matches:
            return 0.0
        
        # Calculate weighted average of match scores
        total_score = 0.0
        total_weight = 0.0
        
        for match in matches:
            weight = match.query_weight * match.doc_weight
            score = match.similarity_score * weight
            total_score += score
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _enhance_with_tfidf(self, text: str, concepts: List[Concept]) -> List[Concept]:
        """Enhance concept weights using TF-IDF analysis"""
        try:
            # Simple TF-IDF enhancement
            words = text.lower().split()
            word_freq = Counter(words)
            total_words = len(words)
            
            enhanced_concepts = []
            for concept in concepts:
                # Boost weight if concept terms appear frequently
                concept_words = concept.name.replace("_", " ").split()
                concept_freq = sum(word_freq.get(word, 0) for word in concept_words)
                
                if concept_freq > 0:
                    tf_boost = min(0.3, concept_freq / total_words * 10)  # Max boost of 0.3
                    new_weight = min(1.0, concept.weight + tf_boost)
                    concept.weight = new_weight
                
                enhanced_concepts.append(concept)
            
            return enhanced_concepts
            
        except Exception as e:
            logger.warning(f"TF-IDF enhancement failed: {str(e)}")
            return concepts
    
    def _fallback_concept_extraction(self, text: str) -> List[Concept]:
        """Fallback concept extraction using rule-based approach"""
        concepts = []
        text_lower = text.lower()
        
        fallback_patterns = {
            "work_hours": (r"(work.*hour|office.*hour|9.*am|5.*pm)", 0.7),
            "remote_work": (r"(remote|work.*home|telework)", 0.6),
            "benefits": (r"(benefit|insurance|401k|health)", 0.8),
            "contact": (r"(contact|phone|email|support)", 0.6),
            "policy": (r"(policy|procedure|rule|guideline)", 0.7)
        }
        
        for concept_name, (pattern, weight) in fallback_patterns.items():
            if re.search(pattern, text_lower):
                concepts.append(Concept(
                    name=concept_name,
                    weight=weight,
                    context=f"Pattern match: {pattern}",
                    category=self._get_concept_category(concept_name)
                ))
        
        return concepts[:5]  # Limit to top 5 concepts
    
    def _calculate_concept_similarity(self, concept1: str, concept2: str) -> float:
        """Calculate similarity between two concept names"""
        try:
            # Simple word-based similarity
            words1 = set(concept1.replace("_", " ").split())
            words2 = set(concept2.replace("_", " ").split())
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _get_concept_category(self, concept_name: str) -> str:
        """Get category for a concept name"""
        for category, concepts in self.concept_categories.items():
            if any(c in concept_name for c in concepts):
                return category
        return "general"
    
    def save_concept_index(self, filepath: str) -> bool:
        """Save concept index to file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dict(self.concept_index), f, indent=2, ensure_ascii=False)
            logger.info(f"Concept index saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving concept index: {str(e)}")
            return False
    
    def load_concept_index(self, filepath: str) -> bool:
        """Load concept index from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.concept_index = defaultdict(dict, json.load(f))
                logger.info(f"Concept index loaded from {filepath}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading concept index: {str(e)}")
            return False