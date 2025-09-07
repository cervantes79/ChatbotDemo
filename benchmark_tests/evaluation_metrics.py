"""
Evaluation Metrics for ICAR vs Traditional RAG Benchmark
Author: Barış Genç
"""

import json
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics

@dataclass
class QueryResult:
    """Single query test result"""
    query_id: str
    query_text: str
    query_type: str
    difficulty: str
    
    # System response
    system_type: str  # "ICAR" or "Traditional_RAG" 
    response_text: str
    retrieval_method: str
    processing_time: float
    
    # Retrieved documents
    retrieved_docs: List[Dict[str, Any]]
    top_doc_source: str
    
    # Evaluation scores
    relevance_score: float = 0.0
    accuracy_score: float = 0.0
    completeness_score: float = 0.0
    concept_match_score: float = 0.0

@dataclass 
class BenchmarkResults:
    """Complete benchmark test results"""
    test_metadata: Dict[str, Any]
    icar_results: List[QueryResult] 
    traditional_rag_results: List[QueryResult]
    comparison_metrics: Dict[str, Any]

class EvaluationMetrics:
    """Evaluation metrics and scoring system"""
    
    def __init__(self):
        self.results = []
        
    def calculate_relevance_score(self, query_result: QueryResult, test_case: Dict[str, Any]) -> float:
        """Calculate how relevant the response is to the query"""
        try:
            response = query_result.response_text.lower()
            expected_terms = test_case.get("ideal_answer_contains", [])
            
            if not expected_terms:
                return 0.5  # Neutral score if no expected terms
            
            # Check for expected terms in response
            found_terms = 0
            for term in expected_terms:
                if term.lower() in response:
                    found_terms += 1
            
            relevance = found_terms / len(expected_terms)
            return min(1.0, relevance)
            
        except Exception as e:
            print(f"Error calculating relevance score: {e}")
            return 0.0
    
    def calculate_accuracy_score(self, query_result: QueryResult, test_case: Dict[str, Any]) -> float:
        """Calculate accuracy based on source document retrieval"""
        try:
            expected_source = test_case.get("expected_source", "")
            retrieved_source = query_result.top_doc_source
            
            if expected_source == "multiple_possible":
                return 0.7  # Moderate score for ambiguous queries
            
            if expected_source.lower() in retrieved_source.lower():
                return 1.0
            
            # Partial credit for related documents
            expected_concepts = test_case.get("expected_concepts", [])
            for concept in expected_concepts:
                if concept.lower() in retrieved_source.lower():
                    return 0.6  # Partial credit
            
            return 0.0
            
        except Exception as e:
            print(f"Error calculating accuracy score: {e}")
            return 0.0
    
    def calculate_completeness_score(self, query_result: QueryResult, test_case: Dict[str, Any]) -> float:
        """Calculate how complete the response is"""
        try:
            response = query_result.response_text
            expected_terms = test_case.get("ideal_answer_contains", [])
            
            if not response or len(response) < 20:
                return 0.0  # Too short
            
            if len(response) > 1000:
                return 0.8  # Might be too verbose
            
            # Check coverage of expected terms
            coverage = 0
            for term in expected_terms:
                if term.lower() in response.lower():
                    coverage += 1
            
            if expected_terms:
                return min(1.0, coverage / len(expected_terms))
            
            return 0.7  # Default moderate score
            
        except Exception as e:
            print(f"Error calculating completeness score: {e}")
            return 0.0
    
    def calculate_concept_match_score(self, query_result: QueryResult, test_case: Dict[str, Any]) -> float:
        """Calculate concept matching effectiveness (mainly for ICAR)"""
        try:
            if query_result.system_type == "Traditional_RAG":
                return 0.0  # Traditional RAG doesn't do concept matching
            
            expected_concepts = test_case.get("expected_concepts", [])
            response = query_result.response_text.lower()
            
            matched_concepts = 0
            for concept in expected_concepts:
                if concept.lower() in response:
                    matched_concepts += 1
            
            if expected_concepts:
                return matched_concepts / len(expected_concepts)
            
            return 0.0
            
        except Exception as e:
            print(f"Error calculating concept match score: {e}")
            return 0.0
    
    def evaluate_query_result(self, query_result: QueryResult, test_case: Dict[str, Any]) -> QueryResult:
        """Evaluate a single query result"""
        query_result.relevance_score = self.calculate_relevance_score(query_result, test_case)
        query_result.accuracy_score = self.calculate_accuracy_score(query_result, test_case)
        query_result.completeness_score = self.calculate_completeness_score(query_result, test_case)
        query_result.concept_match_score = self.calculate_concept_match_score(query_result, test_case)
        
        return query_result
    
    def calculate_aggregate_metrics(self, results: List[QueryResult]) -> Dict[str, Any]:
        """Calculate aggregate metrics for a system"""
        if not results:
            return {}
        
        metrics = {
            "total_queries": len(results),
            "avg_relevance": statistics.mean([r.relevance_score for r in results]),
            "avg_accuracy": statistics.mean([r.accuracy_score for r in results]),
            "avg_completeness": statistics.mean([r.completeness_score for r in results]),
            "avg_concept_match": statistics.mean([r.concept_match_score for r in results]),
            "avg_processing_time": statistics.mean([r.processing_time for r in results]),
        }
        
        # Calculate overall score (weighted average)
        metrics["overall_score"] = (
            metrics["avg_relevance"] * 0.35 +
            metrics["avg_accuracy"] * 0.30 + 
            metrics["avg_completeness"] * 0.20 +
            metrics["avg_concept_match"] * 0.15
        )
        
        # Query type breakdown
        query_types = {}
        for result in results:
            qtype = result.query_type
            if qtype not in query_types:
                query_types[qtype] = []
            query_types[qtype].append(result)
        
        for qtype, qresults in query_types.items():
            metrics[f"{qtype}_count"] = len(qresults)
            metrics[f"{qtype}_avg_score"] = statistics.mean([
                (r.relevance_score + r.accuracy_score + r.completeness_score) / 3 
                for r in qresults
            ])
        
        return metrics
    
    def compare_systems(self, icar_results: List[QueryResult], 
                       traditional_results: List[QueryResult]) -> Dict[str, Any]:
        """Compare ICAR vs Traditional RAG performance"""
        
        icar_metrics = self.calculate_aggregate_metrics(icar_results)
        traditional_metrics = self.calculate_aggregate_metrics(traditional_results)
        
        comparison = {
            "icar_metrics": icar_metrics,
            "traditional_metrics": traditional_metrics,
            "improvements": {}
        }
        
        # Calculate improvements
        if traditional_metrics and icar_metrics:
            comparison["improvements"] = {
                "relevance_improvement": (
                    (icar_metrics["avg_relevance"] - traditional_metrics["avg_relevance"]) / 
                    traditional_metrics["avg_relevance"] * 100
                ) if traditional_metrics["avg_relevance"] > 0 else 0,
                
                "accuracy_improvement": (
                    (icar_metrics["avg_accuracy"] - traditional_metrics["avg_accuracy"]) / 
                    traditional_metrics["avg_accuracy"] * 100
                ) if traditional_metrics["avg_accuracy"] > 0 else 0,
                
                "overall_improvement": (
                    (icar_metrics["overall_score"] - traditional_metrics["overall_score"]) / 
                    traditional_metrics["overall_score"] * 100
                ) if traditional_metrics["overall_score"] > 0 else 0,
                
                "concept_advantage": icar_metrics["avg_concept_match"] * 100  # Traditional is 0
            }
            
            # Winner determination
            comparison["winner"] = "ICAR" if icar_metrics["overall_score"] > traditional_metrics["overall_score"] else "Traditional RAG"
            comparison["performance_gap"] = abs(icar_metrics["overall_score"] - traditional_metrics["overall_score"])
        
        return comparison
    
    def export_results(self, benchmark_results: BenchmarkResults, 
                      filename: str = "benchmark_results.json") -> Path:
        """Export benchmark results to JSON"""
        try:
            results_dict = {
                "metadata": benchmark_results.test_metadata,
                "icar_results": [asdict(r) for r in benchmark_results.icar_results],
                "traditional_rag_results": [asdict(r) for r in benchmark_results.traditional_rag_results],
                "comparison_metrics": benchmark_results.comparison_metrics,
                "summary": self._generate_summary(benchmark_results)
            }
            
            filepath = Path(__file__).parent / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Benchmark results exported: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error exporting results: {e}")
            return None
    
    def _generate_summary(self, benchmark_results: BenchmarkResults) -> Dict[str, Any]:
        """Generate executive summary of results"""
        comparison = benchmark_results.comparison_metrics
        
        summary = {
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_queries_tested": len(benchmark_results.icar_results),
            "winner": comparison.get("winner", "Unknown"),
            "performance_gap": f"{comparison.get('performance_gap', 0):.3f}",
            "key_improvements": {
                "relevance": f"{comparison.get('improvements', {}).get('relevance_improvement', 0):.1f}%",
                "accuracy": f"{comparison.get('improvements', {}).get('accuracy_improvement', 0):.1f}%",
                "overall": f"{comparison.get('improvements', {}).get('overall_improvement', 0):.1f}%"
            },
            "icar_advantages": [
                "Concept-aware matching",
                "Better context understanding", 
                "Multi-level retrieval strategy"
            ],
            "recommendation": "Deploy ICAR" if comparison.get("winner") == "ICAR" else "Consider optimization"
        }
        
        return summary