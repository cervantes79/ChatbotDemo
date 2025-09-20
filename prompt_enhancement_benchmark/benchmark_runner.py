"""
Benchmark Runner for Prompt Enhancement Testing
Executes comprehensive comparison between normal and enhanced prompts
Author: Barış Genç
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

# OpenAI imports
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI library not found. Please install: pip install openai")
    exit(1)

from concept_enhancer import PromptConceptEnhancer
from test_cases import TestCasesDataset
from evaluation_metrics import ResponseEvaluator, BenchmarkReporter, EvaluationResult

class PromptEnhancementBenchmark:
    """
    Main benchmark runner for comparing normal vs enhanced prompts
    """

    def __init__(self, api_key: Optional[str] = None):
        # Setup logging
        self._setup_logging()

        # Initialize OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)

        # Initialize components
        self.concept_enhancer = PromptConceptEnhancer()
        self.test_dataset = TestCasesDataset()
        self.evaluator = ResponseEvaluator()
        self.reporter = BenchmarkReporter()

        # Results storage
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

        self.logger.info("Prompt Enhancement Benchmark initialized successfully")

    def _setup_logging(self):
        """Setup detailed logging"""
        log_dir = Path("results")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'benchmark.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run_single_test(self, test_case: Dict, delay: float = 1.0) -> Optional[EvaluationResult]:
        """Run a single test case comparison"""
        try:
            test_id = test_case["id"]
            original_query = test_case["query"]

            self.logger.info(f"Running test {test_id}: {original_query}")

            # Generate enhanced prompt
            enhanced_prompt = self.concept_enhancer.enhance_prompt(original_query)

            # Get responses from both prompts
            normal_response = self._get_llm_response(original_query)
            time.sleep(delay)  # Rate limiting

            enhanced_response = self._get_llm_response(enhanced_prompt)
            time.sleep(delay)  # Rate limiting

            # Evaluate the responses
            result = self.evaluator.evaluate_response_pair(test_case, normal_response, enhanced_response)
            result.enhanced_prompt = enhanced_prompt

            # Log results
            self.logger.info(f"Test {test_id} completed. Improvement: {result.improvement_percentage:.2f}%")

            return result

        except Exception as e:
            self.logger.error(f"Error in test {test_case.get('id', 'unknown')}: {str(e)}")
            return None

    def _get_llm_response(self, prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 300) -> str:
        """Get response from OpenAI LLM"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful and knowledgeable assistant. Provide clear, accurate, and helpful responses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.error(f"Error getting LLM response: {str(e)}")
            return f"Error: Could not generate response - {str(e)}"

    def run_full_benchmark(self, limit: Optional[int] = None, delay: float = 1.0) -> Dict:
        """Run the complete benchmark suite"""
        self.logger.info("Starting full benchmark suite")

        test_cases = self.test_dataset.get_test_cases()
        if limit:
            test_cases = test_cases[:limit]

        total_tests = len(test_cases)
        self.logger.info(f"Running {total_tests} test cases")

        # Track progress
        successful_tests = 0
        failed_tests = 0

        for i, test_case in enumerate(test_cases, 1):
            self.logger.info(f"Progress: {i}/{total_tests} ({(i/total_tests)*100:.1f}%)")

            result = self.run_single_test(test_case, delay)

            if result:
                self.reporter.add_result(result)
                successful_tests += 1
            else:
                failed_tests += 1

            # Save intermediate results every 10 tests
            if i % 10 == 0:
                self._save_intermediate_results(i)

        # Generate final report
        summary = self.reporter.generate_summary_report()

        self.logger.info(f"Benchmark completed. Successful: {successful_tests}, Failed: {failed_tests}")
        self.logger.info(f"Average improvement: {summary['summary']['avg_improvement']:.2f}%")

        # Save final results
        self._save_final_results(summary)

        return summary

    def run_category_benchmark(self, category: str, delay: float = 1.0) -> Dict:
        """Run benchmark for a specific category"""
        self.logger.info(f"Running benchmark for category: {category}")

        test_cases = self.test_dataset.get_test_cases_by_category(category)
        total_tests = len(test_cases)

        if total_tests == 0:
            self.logger.warning(f"No test cases found for category: {category}")
            return {}

        self.logger.info(f"Running {total_tests} test cases for {category}")

        for i, test_case in enumerate(test_cases, 1):
            self.logger.info(f"Category {category} progress: {i}/{total_tests}")

            result = self.run_single_test(test_case, delay)
            if result:
                self.reporter.add_result(result)

        summary = self.reporter.generate_summary_report()
        self._save_category_results(category, summary)

        return summary

    def _save_intermediate_results(self, test_number: int):
        """Save intermediate results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"intermediate_results_{test_number}_{timestamp}.json"

        try:
            summary = self.reporter.generate_summary_report()
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Intermediate results saved: {filename}")
        except Exception as e:
            self.logger.error(f"Error saving intermediate results: {str(e)}")

    def _save_final_results(self, summary: Dict):
        """Save final benchmark results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary
        summary_file = self.results_dir / f"benchmark_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save detailed results
        detailed_file = self.results_dir / f"benchmark_detailed_{timestamp}.json"
        self.reporter.save_detailed_results(str(detailed_file))

        # Generate markdown report
        self._generate_markdown_report(summary, timestamp)

        self.logger.info(f"Final results saved:")
        self.logger.info(f"  Summary: {summary_file}")
        self.logger.info(f"  Detailed: {detailed_file}")

    def _save_category_results(self, category: str, summary: Dict):
        """Save category-specific results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"benchmark_{category}_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Category {category} results saved: {filename}")

    def _generate_markdown_report(self, summary: Dict, timestamp: str):
        """Generate a markdown report"""
        report_file = self.results_dir / f"benchmark_report_{timestamp}.md"

        report_content = f"""# Prompt Enhancement Benchmark Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Methodology:** ICAR Concept Enhancement vs Normal Prompts
**Author:** Barış Genç

## Executive Summary

- **Total Tests:** {summary['summary']['total_tests']}
- **Average Improvement:** {summary['summary']['avg_improvement']:.2f}%
- **Success Rate:** {summary['summary']['success_rate']:.2f}%
- **Positive Improvements:** {summary['summary']['positive_improvements']} out of {summary['summary']['total_tests']}

## Metric Analysis

"""

        for metric, data in summary['metric_analysis'].items():
            report_content += f"""### {metric.title()} Metric
- **Normal Average:** {data['avg_normal']:.3f}
- **Enhanced Average:** {data['avg_enhanced']:.3f}
- **Improvement:** {data['improvement_percent']:.2f}%

"""

        report_content += """## Category Performance

"""

        for category, data in summary['category_analysis'].items():
            report_content += f"""### {category.title()} Category
- **Test Count:** {data['count']}
- **Average Improvement:** {data['avg_improvement']:.2f}%
- **Success Rate:** {(data['positive_improvements']/data['count']*100):.1f}%

"""

        report_content += f"""## Methodology

This benchmark compares the effectiveness of ICAR concept-enhanced prompts against normal prompts without any RAG retrieval. The enhancement process extracts key concepts from queries and provides contextual understanding to the LLM.

**Test Categories:**
- Business queries
- Technical questions
- Weather requests
- General knowledge
- Complex reasoning

**Evaluation Metrics:**
- Relevance to query
- Concept understanding
- Response quality
- Specificity
- Completeness

## Conclusion

{'The ICAR concept enhancement methodology shows significant improvements in prompt effectiveness.' if summary['summary']['avg_improvement'] > 10 else 'The concept enhancement shows modest improvements that may warrant further investigation.'}

---
*Generated by ICAR Prompt Enhancement Benchmark*
*Author: Barış Genç*
"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"Markdown report generated: {report_file}")

    def get_concept_analysis_for_query(self, query: str) -> Dict:
        """Get detailed concept analysis for a single query"""
        return self.concept_enhancer.get_concept_summary(query)

    def test_enhancement_preview(self, query: str) -> Dict:
        """Preview enhancement for a single query without LLM calls"""
        original = query
        enhanced = self.concept_enhancer.enhance_prompt(query)
        analysis = self.concept_enhancer.get_concept_summary(query)

        return {
            "original_query": original,
            "enhanced_prompt": enhanced,
            "concept_analysis": analysis
        }


def main():
    """Main execution function for CLI usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Run Prompt Enhancement Benchmark")
    parser.add_argument("--limit", type=int, help="Limit number of test cases")
    parser.add_argument("--category", type=str, help="Run tests for specific category only")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (seconds)")
    parser.add_argument("--preview", type=str, help="Preview enhancement for a single query")

    args = parser.parse_args()

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    try:
        benchmark = PromptEnhancementBenchmark()

        if args.preview:
            # Preview mode
            result = benchmark.test_enhancement_preview(args.preview)
            print(f"\nOriginal Query: {result['original_query']}")
            print(f"\nEnhanced Prompt:\n{result['enhanced_prompt']}")
            print(f"\nConcept Analysis: {json.dumps(result['concept_analysis'], indent=2)}")
            return

        if args.category:
            # Category-specific benchmark
            summary = benchmark.run_category_benchmark(args.category, args.delay)
        else:
            # Full benchmark
            summary = benchmark.run_full_benchmark(args.limit, args.delay)

        print(f"\nBenchmark Results:")
        print(f"Average Improvement: {summary['summary']['avg_improvement']:.2f}%")
        print(f"Success Rate: {summary['summary']['success_rate']:.2f}%")

    except Exception as e:
        print(f"Error running benchmark: {str(e)}")


if __name__ == "__main__":
    main()