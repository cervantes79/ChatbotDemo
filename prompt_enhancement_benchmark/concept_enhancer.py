"""
ICAR Concept Enhancement for Prompts
Standalone concept extraction without RAG dependency
Author: Barış Genç
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Concept:
    name: str
    weight: float
    category: str = "general"
    confidence: float = 1.0

class PromptConceptEnhancer:
    """
    Extract concepts from prompts and enhance them for better LLM understanding
    Based on ICAR methodology but without vector store dependency
    """

    def __init__(self):
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
            ],
            "location": [
                "address", "location", "place", "city", "country", "street", "building",
                "map", "directions", "distance", "travel", "transportation"
            ],
            "time": [
                "time", "date", "schedule", "calendar", "appointment", "deadline",
                "duration", "period", "frequency", "timing", "when", "until"
            ],
            "financial": [
                "money", "cost", "price", "payment", "budget", "invoice", "bill",
                "discount", "tax", "profit", "expense", "financial", "banking"
            ]
        }

        # Weather-specific patterns
        self.weather_patterns = [
            r"weather in (\w+)",
            r"temperature in (\w+)",
            r"forecast for (\w+)",
            r"what's the weather",
            r"how's the weather",
            r"weather like in"
        ]

    def extract_concepts(self, text: str) -> List[Concept]:
        """Extract concepts from text"""
        text_lower = text.lower()
        concepts = []

        # Extract concepts by category
        for category, keywords in self.concept_categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Calculate weight based on keyword importance and frequency
                    weight = text_lower.count(keyword) * self._get_keyword_importance(keyword)
                    confidence = self._calculate_confidence(keyword, text_lower)

                    concepts.append(Concept(
                        name=keyword,
                        weight=weight,
                        category=category,
                        confidence=confidence
                    ))

        # Check for weather location patterns
        for pattern in self.weather_patterns:
            if re.search(pattern, text_lower):
                concepts.append(Concept(
                    name="weather_query",
                    weight=3.0,
                    category="weather",
                    confidence=0.9
                ))

        # Remove duplicates and sort by weight
        unique_concepts = {}
        for concept in concepts:
            if concept.name not in unique_concepts or concept.weight > unique_concepts[concept.name].weight:
                unique_concepts[concept.name] = concept

        return sorted(unique_concepts.values(), key=lambda x: x.weight, reverse=True)

    def _get_keyword_importance(self, keyword: str) -> float:
        """Assign importance weights to different keywords"""
        high_importance = ["weather", "temperature", "policy", "employee", "system", "algorithm"]
        medium_importance = ["work", "data", "meeting", "project", "feature"]

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

    def enhance_prompt(self, original_prompt: str) -> str:
        """Enhance prompt with concept analysis"""
        concepts = self.extract_concepts(original_prompt)

        if not concepts:
            return original_prompt

        # Get primary category
        primary_category = self._get_primary_category(concepts)

        # Build enhancement text
        concept_names = [c.name for c in concepts[:5]]  # Top 5 concepts
        high_confidence_concepts = [c.name for c in concepts if c.confidence > 0.7]

        enhancement = f"""[CONCEPT ANALYSIS]
Primary Category: {primary_category}
Key Concepts: {', '.join(concept_names)}
High Confidence Concepts: {', '.join(high_confidence_concepts)}
Context Understanding: This query relates to {primary_category} domain and involves concepts like {', '.join(concept_names[:3])}.

[ORIGINAL QUERY]
{original_prompt}

[ENHANCED CONTEXT]
Please provide a response that considers the {primary_category} context and addresses the key concepts: {', '.join(concept_names[:3])}."""

        return enhancement

    def _get_primary_category(self, concepts: List[Concept]) -> str:
        """Determine the primary category from concepts"""
        if not concepts:
            return "general"

        # Count concepts by category
        category_scores = {}
        for concept in concepts:
            if concept.category not in category_scores:
                category_scores[concept.category] = 0
            category_scores[concept.category] += concept.weight * concept.confidence

        if not category_scores:
            return "general"

        return max(category_scores.items(), key=lambda x: x[1])[0]

    def get_concept_summary(self, text: str) -> Dict:
        """Get detailed concept analysis summary"""
        concepts = self.extract_concepts(text)
        primary_category = self._get_primary_category(concepts)

        return {
            "original_text": text,
            "primary_category": primary_category,
            "concepts_found": len(concepts),
            "top_concepts": [
                {
                    "name": c.name,
                    "category": c.category,
                    "weight": c.weight,
                    "confidence": c.confidence
                } for c in concepts[:5]
            ],
            "category_distribution": self._get_category_distribution(concepts)
        }

    def _get_category_distribution(self, concepts: List[Concept]) -> Dict[str, int]:
        """Get distribution of concepts across categories"""
        distribution = {}
        for concept in concepts:
            if concept.category not in distribution:
                distribution[concept.category] = 0
            distribution[concept.category] += 1
        return distribution