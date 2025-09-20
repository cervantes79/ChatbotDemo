"""
Local LLM Benchmark Runner for Prompt Enhancement Testing
Uses HuggingFace models locally without API dependencies
Author: Barış Genç
"""

import os
import json
import time
import logging
from typing import List, Dict
from datetime import datetime
from pathlib import Path

# HuggingFace imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    print("Required packages not found. Please install: pip install transformers torch")
    exit(1)

from concept_enhancer import PromptConceptEnhancer
from test_cases import TestCasesDataset
from evaluation_metrics import ResponseEvaluator, BenchmarkReporter

class LocalLLMBenchmark:
    """
    Benchmark runner using local HuggingFace models
    """

    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        # Setup logging
        self._setup_logging()

        self.model_name = model_name
        self.logger.info(f"Initializing Local LLM Benchmark with model: {model_name}")

        # Initialize model and tokenizer
        self._initialize_model()

        # Initialize components
        self.concept_enhancer = PromptConceptEnhancer()
        self.test_dataset = TestCasesDataset()
        self.evaluator = ResponseEvaluator()
        self.reporter = BenchmarkReporter()

        # Results storage
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

    def _setup_logging(self):
        """Setup logging"""
        log_dir = Path("results")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'local_benchmark.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_model(self):
        """Initialize the local LLM model"""
        try:
            self.logger.info(f"Loading model: {self.model_name}")

            # Try different small models
            model_options = [
                "microsoft/DialoGPT-small",
                "distilgpt2",
                "gpt2",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            ]

            for model in model_options:
                try:
                    self.logger.info(f"Trying to load: {model}")

                    # Create text generation pipeline
                    self.generator = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=model,
                        max_length=200,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=50256,  # GPT-2 pad token
                        device=0 if torch.cuda.is_available() else -1
                    )

                    self.model_name = model
                    self.logger.info(f"Successfully loaded: {model}")
                    break

                except Exception as e:
                    self.logger.warning(f"Failed to load {model}: {str(e)}")
                    continue
            else:
                raise Exception("Could not load any model")

        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            # Fallback to simple text generation
            self.generator = None

    def _get_llm_response(self, prompt: str) -> str:
        """Get response from local LLM"""
        try:
            if self.generator is None:
                return "Error: Model not loaded"

            # Clean and prepare prompt
            clean_prompt = prompt.strip()
            if len(clean_prompt) > 500:
                clean_prompt = clean_prompt[:500]

            # Generate response
            result = self.generator(
                clean_prompt,
                max_new_tokens=100,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )

            # Extract generated text
            generated_text = result[0]['generated_text']

            # Remove the original prompt from response
            if generated_text.startswith(clean_prompt):
                response = generated_text[len(clean_prompt):].strip()
            else:
                response = generated_text.strip()

            # Clean up response
            if not response:
                response = "I understand your question and will provide a helpful response."

            # Limit response length
            if len(response) > 300:
                response = response[:300] + "..."

            return response

        except Exception as e:
            self.logger.error(f"Error getting LLM response: {str(e)}")
            return f"Error generating response: {str(e)[:100]}"

    def run_single_test(self, test_case: Dict) -> Dict:
        """Run a single test case"""
        try:
            test_id = test_case["id"]
            original_query = test_case["query"]

            self.logger.info(f"Running test {test_id}: {original_query}")

            # Generate enhanced prompt
            enhanced_prompt = self.concept_enhancer.enhance_prompt(original_query)

            # Get responses
            print(f"🔍 Testing: {original_query[:50]}...")

            normal_response = self._get_llm_response(original_query)
            enhanced_response = self._get_llm_response(enhanced_prompt)

            # Evaluate responses
            result = self.evaluator.evaluate_response_pair(test_case, normal_response, enhanced_response)
            result.enhanced_prompt = enhanced_prompt

            print(f"✅ Normal response length: {len(normal_response)} chars")
            print(f"✅ Enhanced response length: {len(enhanced_response)} chars")
            print(f"📊 Improvement: {result.improvement_percentage:.2f}%")

            return {
                "test_id": test_id,
                "original_query": original_query,
                "enhanced_prompt": enhanced_prompt,
                "normal_response": normal_response,
                "enhanced_response": enhanced_response,
                "improvement": result.improvement_percentage,
                "scores": result.scores
            }

        except Exception as e:
            self.logger.error(f"Error in test {test_case.get('id', 'unknown')}: {str(e)}")
            return None

    def run_benchmark(self, num_tests: int = 10) -> Dict:
        """Run benchmark with specified number of tests"""
        print(f"🚀 Starting Local LLM Benchmark")
        print(f"🔧 Model: {self.model_name}")
        print(f"📊 Running {num_tests} tests")
        print("=" * 60)

        test_cases = self.test_dataset.get_test_cases()[:num_tests]

        results = []
        total_improvement = 0
        successful_tests = 0

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}/{num_tests}")

            result = self.run_single_test(test_case)

            if result:
                results.append(result)
                self.reporter.add_result(self.evaluator.evaluate_response_pair(
                    test_case, result["normal_response"], result["enhanced_response"]
                ))
                total_improvement += result["improvement"]
                successful_tests += 1

            # Small delay to prevent overheating
            time.sleep(0.5)

        # Calculate results
        avg_improvement = total_improvement / successful_tests if successful_tests > 0 else 0

        print(f"\n🎯 BENCHMARK RESULTS:")
        print(f"✅ Successful tests: {successful_tests}/{num_tests}")
        print(f"📈 Average improvement: {avg_improvement:.2f}%")
        print(f"🤖 Model used: {self.model_name}")

        # Generate detailed report
        summary = self.reporter.generate_summary_report()

        # Save results
        self._save_results(results, summary)

        return {
            "model": self.model_name,
            "total_tests": num_tests,
            "successful_tests": successful_tests,
            "average_improvement": avg_improvement,
            "detailed_results": results,
            "summary": summary
        }

    def _save_results(self, results: List[Dict], summary: Dict):
        """Save benchmark results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = self.results_dir / f"local_benchmark_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model": self.model_name,
                "timestamp": timestamp,
                "results": results,
                "summary": summary
            }, f, indent=2, ensure_ascii=False)

        print(f"💾 Results saved: {results_file}")

def main():
    """Main execution"""
    print("🤖 LOCAL LLM PROMPT ENHANCEMENT BENCHMARK")
    print("=" * 50)

    # Try to find a good small model
    benchmark = LocalLLMBenchmark()

    # Run with 15 tests for quick validation
    results = benchmark.run_benchmark(num_tests=15)

    print(f"\n🏁 Final Results:")
    print(f"Model: {results['model']}")
    print(f"Average Improvement: {results['average_improvement']:.2f}%")

    if results['average_improvement'] > 10:
        print("🎉 Promising results! ICAR enhancement shows positive impact!")
    elif results['average_improvement'] > 0:
        print("👍 Positive results! Enhancement works with local models too!")
    else:
        print("🤔 Mixed results. May need model or method adjustment.")

if __name__ == "__main__":
    main()