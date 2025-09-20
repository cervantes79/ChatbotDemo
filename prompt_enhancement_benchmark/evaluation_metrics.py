"""
Evaluation Metrics for Prompt Enhancement Benchmark
Comprehensive scoring system to compare normal vs enhanced prompts
Author: Barış Genç
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    test_id: str
    original_prompt: str
    enhanced_prompt: str
    normal_response: str
    enhanced_response: str
    scores: Dict[str, float]
    improvement_percentage: float
    category: str
    timestamp: str

class ResponseEvaluator:
    """
    Comprehensive evaluation system for comparing normal vs enhanced responses
    """

    def __init__(self):
        self.category_keywords = {
            "business": ["company", "employee", "work", "policy", "procedure", "office", "team", "manager"],
            "technical": ["system", "algorithm", "data", "software", "programming", "development", "security"],
            "weather": ["weather", "temperature", "forecast", "rain", "sunny", "cloudy", "storm"],
            "general": ["knowledge", "fact", "information", "answer", "explanation"],
            "complex": ["analysis", "strategy", "implication", "effect", "impact", "solution", "approach"]
        }

    def evaluate_response_pair(self, test_case: Dict, normal_response: str, enhanced_response: str) -> EvaluationResult:
        """Evaluate a pair of responses (normal vs enhanced)"""

        scores = {}

        # 1. Relevance Score
        scores["relevance_normal"] = self._calculate_relevance(test_case, normal_response)
        scores["relevance_enhanced"] = self._calculate_relevance(test_case, enhanced_response)

        # 2. Concept Understanding Score
        scores["concept_normal"] = self._calculate_concept_understanding(test_case, normal_response)
        scores["concept_enhanced"] = self._calculate_concept_understanding(test_case, enhanced_response)

        # 3. Response Quality Score
        scores["quality_normal"] = self._calculate_response_quality(normal_response)
        scores["quality_enhanced"] = self._calculate_response_quality(enhanced_response)

        # 4. Specificity Score
        scores["specificity_normal"] = self._calculate_specificity(test_case, normal_response)
        scores["specificity_enhanced"] = self._calculate_specificity(test_case, enhanced_response)

        # 5. Completeness Score
        scores["completeness_normal"] = self._calculate_completeness(test_case, normal_response)
        scores["completeness_enhanced"] = self._calculate_completeness(test_case, enhanced_response)

        # Calculate overall improvement
        improvement = self._calculate_improvement(scores)

        return EvaluationResult(
            test_id=test_case["id"],
            original_prompt=test_case["query"],
            enhanced_prompt="",  # Will be filled by benchmark runner
            normal_response=normal_response,
            enhanced_response=enhanced_response,
            scores=scores,
            improvement_percentage=improvement,
            category=test_case["category"],
            timestamp=datetime.now().isoformat()
        )

    def _calculate_relevance(self, test_case: Dict, response: str) -> float:
        """Calculate how relevant the response is to the query"""
        query = test_case["query"].lower()
        response = response.lower()

        # Check for query keywords in response
        query_words = set(query.split())
        response_words = set(response.split())

        # Remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "what", "how", "when", "where", "why", "who"}

        query_words = query_words - common_words
        response_words = response_words - common_words

        if not query_words:
            return 0.5

        # Calculate overlap
        overlap = len(query_words.intersection(response_words))
        relevance_score = overlap / len(query_words)

        # Bonus for expected concepts
        expected_concepts = test_case.get("expected_concepts", [])
        concept_matches = sum(1 for concept in expected_concepts if concept.lower() in response)
        concept_bonus = (concept_matches / len(expected_concepts)) * 0.3 if expected_concepts else 0

        return min(1.0, relevance_score + concept_bonus)

    def _calculate_concept_understanding(self, test_case: Dict, response: str) -> float:
        """Calculate how well the response demonstrates understanding of key concepts"""
        category = test_case["category"]
        expected_concepts = test_case.get("expected_concepts", [])
        response_lower = response.lower()

        concept_score = 0.0

        # Check for expected concepts
        if expected_concepts:
            found_concepts = sum(1 for concept in expected_concepts if concept.lower() in response_lower)
            concept_score += (found_concepts / len(expected_concepts)) * 0.6

        # Check for category-related keywords
        category_keywords = self.category_keywords.get(category, [])
        if category_keywords:
            found_keywords = sum(1 for keyword in category_keywords if keyword in response_lower)
            keyword_score = min(0.4, (found_keywords / len(category_keywords)) * 0.4)
            concept_score += keyword_score

        # Length and detail bonus (more detailed responses often show better understanding)
        if len(response.split()) > 20:
            concept_score += 0.1
        if len(response.split()) > 50:
            concept_score += 0.1

        return min(1.0, concept_score)

    def _calculate_response_quality(self, response: str) -> float:
        """Calculate overall response quality"""
        if not response or len(response.strip()) < 10:
            return 0.0

        quality_score = 0.0

        # Length appropriateness (not too short, not too long)
        word_count = len(response.split())
        if 15 <= word_count <= 200:
            quality_score += 0.3
        elif 10 <= word_count <= 300:
            quality_score += 0.2
        else:
            quality_score += 0.1

        # Sentence structure
        sentences = response.split('.')
        if len(sentences) >= 2:
            quality_score += 0.2

        # Avoid repetition
        words = response.lower().split()
        unique_words = set(words)
        if len(unique_words) / len(words) > 0.7:
            quality_score += 0.2

        # Professional tone indicators
        professional_indicators = ["however", "therefore", "additionally", "furthermore", "consequently", "moreover"]
        if any(indicator in response.lower() for indicator in professional_indicators):
            quality_score += 0.1

        # Helpful structure indicators
        structure_indicators = ["first", "second", "finally", "in conclusion", "for example", "such as"]
        if any(indicator in response.lower() for indicator in structure_indicators):
            quality_score += 0.2

        return min(1.0, quality_score)

    def _calculate_specificity(self, test_case: Dict, response: str) -> float:
        """Calculate how specific and detailed the response is"""
        if not response:
            return 0.0

        specificity_score = 0.0
        response_lower = response.lower()

        # Check for specific details (numbers, dates, names, etc.)
        if re.search(r'\d+', response):
            specificity_score += 0.3

        # Check for specific examples
        if any(phrase in response_lower for phrase in ["for example", "such as", "including", "specifically"]):
            specificity_score += 0.2

        # Check for technical terms or domain-specific language
        category = test_case["category"]
        category_keywords = self.category_keywords.get(category, [])
        technical_words = sum(1 for word in category_keywords if word in response_lower)
        if technical_words > 0:
            specificity_score += min(0.3, technical_words * 0.1)

        # Length-based specificity
        word_count = len(response.split())
        if word_count > 30:
            specificity_score += 0.2

        return min(1.0, specificity_score)

    def _calculate_completeness(self, test_case: Dict, response: str) -> float:
        """Calculate how complete the response is"""
        if not response:
            return 0.0

        completeness_score = 0.0

        # Check if response addresses the main question
        query_type = self._identify_query_type(test_case["query"])
        if self._addresses_query_type(query_type, response):
            completeness_score += 0.4

        # Check for comprehensive coverage of expected concepts
        expected_concepts = test_case.get("expected_concepts", [])
        if expected_concepts:
            covered_concepts = sum(1 for concept in expected_concepts if concept.lower() in response.lower())
            coverage_ratio = covered_concepts / len(expected_concepts)
            completeness_score += coverage_ratio * 0.4

        # Check for conclusion or summary
        if any(phrase in response.lower() for phrase in ["in conclusion", "to summary", "overall", "in summary"]):
            completeness_score += 0.1

        # Length adequacy
        word_count = len(response.split())
        if word_count >= 20:
            completeness_score += 0.1

        return min(1.0, completeness_score)

    def _identify_query_type(self, query: str) -> str:
        """Identify the type of query (what, how, why, etc.)"""
        query_lower = query.lower()
        if query_lower.startswith(("what", "what's")):
            return "what"
        elif query_lower.startswith("how"):
            return "how"
        elif query_lower.startswith("why"):
            return "why"
        elif query_lower.startswith(("when", "where", "who")):
            return "factual"
        elif "?" in query:
            return "question"
        else:
            return "statement"

    def _addresses_query_type(self, query_type: str, response: str) -> bool:
        """Check if response appropriately addresses the query type"""
        response_lower = response.lower()

        if query_type == "what":
            return any(word in response_lower for word in ["is", "are", "means", "refers", "definition"])
        elif query_type == "how":
            return any(word in response_lower for word in ["by", "through", "using", "steps", "process", "method"])
        elif query_type == "why":
            return any(word in response_lower for word in ["because", "due to", "reason", "cause", "since"])
        elif query_type == "factual":
            return len(response.split()) >= 5  # Factual answers should have some substance
        else:
            return True

    def _calculate_improvement(self, scores: Dict[str, float]) -> float:
        """Calculate overall improvement percentage"""
        metrics = ["relevance", "concept", "quality", "specificity", "completeness"]

        total_normal = sum(scores[f"{metric}_normal"] for metric in metrics)
        total_enhanced = sum(scores[f"{metric}_enhanced"] for metric in metrics)

        if total_normal == 0:
            return 0.0

        improvement = ((total_enhanced - total_normal) / total_normal) * 100
        return round(improvement, 2)

class BenchmarkReporter:
    """Generate comprehensive benchmark reports"""

    def __init__(self):
        self.results = []

    def add_result(self, result: EvaluationResult):
        """Add an evaluation result"""
        self.results.append(result)

    def generate_summary_report(self) -> Dict:
        """Generate summary statistics"""
        if not self.results:
            return {}

        total_tests = len(self.results)
        improvements = [r.improvement_percentage for r in self.results]

        # Calculate average improvements by metric
        metric_improvements = {}
        metrics = ["relevance", "concept", "quality", "specificity", "completeness"]

        for metric in metrics:
            normal_scores = [r.scores[f"{metric}_normal"] for r in self.results]
            enhanced_scores = [r.scores[f"{metric}_enhanced"] for r in self.results]

            avg_normal = sum(normal_scores) / len(normal_scores)
            avg_enhanced = sum(enhanced_scores) / len(enhanced_scores)

            if avg_normal > 0:
                improvement = ((avg_enhanced - avg_normal) / avg_normal) * 100
            else:
                improvement = 0

            metric_improvements[metric] = {
                "avg_normal": round(avg_normal, 3),
                "avg_enhanced": round(avg_enhanced, 3),
                "improvement_percent": round(improvement, 2)
            }

        # Category-wise analysis
        category_stats = {}
        for result in self.results:
            cat = result.category
            if cat not in category_stats:
                category_stats[cat] = []
            category_stats[cat].append(result.improvement_percentage)

        category_summary = {}
        for cat, improvements in category_stats.items():
            category_summary[cat] = {
                "count": len(improvements),
                "avg_improvement": round(sum(improvements) / len(improvements), 2),
                "positive_improvements": len([i for i in improvements if i > 0])
            }

        return {
            "summary": {
                "total_tests": total_tests,
                "avg_improvement": round(sum(improvements) / len(improvements), 2),
                "positive_improvements": len([i for i in improvements if i > 0]),
                "success_rate": round((len([i for i in improvements if i > 0]) / total_tests) * 100, 2)
            },
            "metric_analysis": metric_improvements,
            "category_analysis": category_summary,
            "timestamp": datetime.now().isoformat()
        }

    def save_detailed_results(self, filepath: str):
        """Save detailed results to JSON file"""
        detailed_data = {
            "metadata": {
                "total_tests": len(self.results),
                "timestamp": datetime.now().isoformat(),
                "methodology": "ICAR Prompt Enhancement Benchmark"
            },
            "summary": self.generate_summary_report(),
            "detailed_results": [
                {
                    "test_id": r.test_id,
                    "category": r.category,
                    "original_prompt": r.original_prompt,
                    "improvement_percentage": r.improvement_percentage,
                    "scores": r.scores,
                    "response_lengths": {
                        "normal": len(r.normal_response.split()),
                        "enhanced": len(r.enhanced_response.split())
                    }
                } for r in self.results
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)