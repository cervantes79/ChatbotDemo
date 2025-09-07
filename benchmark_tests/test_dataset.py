"""
ICAR vs Traditional RAG Benchmark Test Dataset
Author: Barış Genç
"""

import json
from typing import List, Dict, Any
from pathlib import Path

class BenchmarkDataset:
    """Generate and manage benchmark test data for ICAR vs RAG comparison"""
    
    def __init__(self):
        self.test_questions = []
        self.expected_sources = []
        self.difficulty_levels = []
    
    def generate_synthetic_dataset(self) -> List[Dict[str, Any]]:
        """Generate comprehensive test dataset covering various query types"""
        
        test_cases = [
            # CONCEPT-HEAVY QUERIES (Where ICAR should excel)
            {
                "question": "What are the company work hours and policies?",
                "expected_concepts": ["work", "hours", "policy", "company"],
                "expected_source": "company_handbook",
                "difficulty": "medium",
                "query_type": "concept_based",
                "ideal_answer_contains": ["Monday", "Friday", "9:00", "5:00"]
            },
            {
                "question": "Tell me about employee benefits and healthcare",
                "expected_concepts": ["employee", "benefits", "healthcare", "insurance"],
                "expected_source": "hr_benefits",
                "difficulty": "medium",
                "query_type": "concept_based",
                "ideal_answer_contains": ["health", "insurance", "coverage", "benefits"]
            },
            {
                "question": "What are the product specifications and features?",
                "expected_concepts": ["product", "specification", "features", "technical"],
                "expected_source": "product_catalog",
                "difficulty": "hard",
                "query_type": "concept_based",
                "ideal_answer_contains": ["specifications", "features", "technical", "product"]
            },
            
            # SEMANTIC QUERIES (Traditional RAG territory)
            {
                "question": "How do I return a defective item?",
                "expected_concepts": ["return", "defective", "item", "policy"],
                "expected_source": "return_policy",
                "difficulty": "easy",
                "query_type": "semantic",
                "ideal_answer_contains": ["return", "refund", "exchange", "policy"]
            },
            {
                "question": "What is the warranty period for products?",
                "expected_concepts": ["warranty", "period", "product", "coverage"],
                "expected_source": "warranty_terms",
                "difficulty": "easy",
                "query_type": "semantic",
                "ideal_answer_contains": ["warranty", "months", "coverage", "terms"]
            },
            
            # COMPLEX QUERIES (Multi-concept matching)
            {
                "question": "How does the IT support process work for remote employees?",
                "expected_concepts": ["IT", "support", "process", "remote", "employee"],
                "expected_source": "it_support_guide",
                "difficulty": "hard",
                "query_type": "complex",
                "ideal_answer_contains": ["support", "remote", "process", "IT"]
            },
            {
                "question": "What training programs are available for new managers?",
                "expected_concepts": ["training", "program", "manager", "new", "development"],
                "expected_source": "training_catalog",
                "difficulty": "hard",
                "query_type": "complex",
                "ideal_answer_contains": ["training", "management", "program", "development"]
            },
            
            # EXACT MATCH QUERIES (Baseline comparison)
            {
                "question": "What is the company phone number?",
                "expected_concepts": ["phone", "number", "contact"],
                "expected_source": "contact_info",
                "difficulty": "easy",
                "query_type": "exact",
                "ideal_answer_contains": ["phone", "number", "contact", "call"]
            },
            {
                "question": "When was the company founded?",
                "expected_concepts": ["founded", "company", "history"],
                "expected_source": "company_history",
                "difficulty": "easy",
                "query_type": "exact",
                "ideal_answer_contains": ["founded", "established", "year", "history"]
            },
            
            # AMBIGUOUS QUERIES (Concept disambiguation test)
            {
                "question": "Tell me about security",
                "expected_concepts": ["security"],
                "expected_source": "multiple_possible",
                "difficulty": "hard",
                "query_type": "ambiguous",
                "ideal_answer_contains": ["security", "policy", "system"]
            },
            {
                "question": "How do I get access?",
                "expected_concepts": ["access"],
                "expected_source": "multiple_possible",
                "difficulty": "medium",
                "query_type": "ambiguous",
                "ideal_answer_contains": ["access", "permission", "login"]
            }
        ]
        
        # Add metadata to each test case
        for i, case in enumerate(test_cases):
            case["id"] = f"test_{i+1:03d}"
            case["category"] = self._categorize_query(case["query_type"])
            
        return test_cases
    
    def _categorize_query(self, query_type: str) -> str:
        """Categorize queries for analysis"""
        categories = {
            "concept_based": "ICAR Advantage",
            "semantic": "Traditional RAG",
            "complex": "ICAR Advantage", 
            "exact": "Baseline",
            "ambiguous": "Disambiguation"
        }
        return categories.get(query_type, "Unknown")
    
    def save_dataset(self, filename: str = "benchmark_dataset.json"):
        """Save test dataset to file"""
        dataset = self.generate_synthetic_dataset()
        
        benchmark_data = {
            "metadata": {
                "description": "ICAR vs Traditional RAG Benchmark Dataset",
                "author": "Barış Genç",
                "total_questions": len(dataset),
                "categories": {
                    "concept_based": len([q for q in dataset if q["query_type"] == "concept_based"]),
                    "semantic": len([q for q in dataset if q["query_type"] == "semantic"]),
                    "complex": len([q for q in dataset if q["query_type"] == "complex"]),
                    "exact": len([q for q in dataset if q["query_type"] == "exact"]),
                    "ambiguous": len([q for q in dataset if q["query_type"] == "ambiguous"])
                }
            },
            "test_cases": dataset
        }
        
        filepath = Path(__file__).parent / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Benchmark dataset saved: {filepath}")
        print(f"   Total test cases: {len(dataset)}")
        print(f"   Categories: {benchmark_data['metadata']['categories']}")
        
        return filepath

if __name__ == "__main__":
    dataset = BenchmarkDataset()
    dataset.save_dataset()