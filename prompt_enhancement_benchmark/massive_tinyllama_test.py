"""
MASSIVE 1000 PROMPT TEST WITH TINYLLAMA 1.1B
ICAR vs NO-ICAR Comprehensive Comparison
Author: Barış Genç
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

from concept_enhancer import PromptConceptEnhancer
from evaluation_metrics import ResponseEvaluator

class MassiveTinyLlamaTest:
    """1000 prompt test with TinyLlama 1.1B"""

    def __init__(self):
        self.setup_logging()
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.concept_enhancer = PromptConceptEnhancer()
        self.evaluator = ResponseEvaluator()
        self.generator = None
        self.results_dir = Path("results2")
        self.results_dir.mkdir(exist_ok=True)

    def setup_logging(self):
        """Setup detailed logging"""
        log_file = Path("results2") / f"massive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        print(f"📋 Detailed logging: {log_file}")

    def load_model(self):
        """Load TinyLlama 1.1B model"""
        try:
            print("🚀 LOADING TINYLLAMA 1.1B MODEL")
            print("🧠 1.1 BILLION PARAMETERS!")
            print("⏳ Loading may take several minutes...")

            self.logger.info(f"Loading TinyLlama model: {self.model_name}")

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )

            # Create pipeline
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=120,
                temperature=0.7,
                do_sample=True,
                device=0 if torch.cuda.is_available() else -1
            )

            print("✅ TinyLlama 1.1B loaded successfully!")
            self.logger.info("TinyLlama model loaded successfully")
            return True

        except Exception as e:
            print(f"❌ Error loading TinyLlama: {str(e)}")
            self.logger.error(f"Error loading model: {str(e)}")
            return False

    def load_test_dataset(self) -> List[Dict]:
        """Load 1000 test prompts"""
        try:
            with open("results2/massive_test_dataset_1000.json", 'r', encoding='utf-8') as f:
                data = json.load(f)

            prompts = data["prompts"]
            print(f"📊 Loaded {len(prompts)} test prompts")
            self.logger.info(f"Loaded {len(prompts)} test prompts")

            return prompts

        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            self.logger.error(f"Error loading dataset: {str(e)}")
            return []

    def generate_response(self, prompt: str, is_enhanced: bool = False) -> str:
        """Generate response with TinyLlama"""
        try:
            # Format for chat
            if is_enhanced:
                chat_prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{prompt}\n<|assistant|>\n"
            else:
                chat_prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{prompt}\n<|assistant|>\n"

            # Generate
            result = self.generator(
                chat_prompt,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )

            # Extract response
            generated_text = result[0]['generated_text']
            if "<|assistant|>" in generated_text:
                response = generated_text.split("<|assistant|>")[-1].strip()
            else:
                response = generated_text.strip()

            # Clean response
            if not response or len(response) < 5:
                response = "I understand your question and will provide a helpful response."

            # Limit length
            if len(response) > 400:
                response = response[:400] + "..."

            return response

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return f"Error: {str(e)[:50]}"

    def run_single_test(self, test_data: Dict, test_num: int, total_tests: int) -> Dict:
        """Run single prompt test"""
        try:
            prompt_id = test_data["id"]
            original_prompt = test_data["prompt"]
            category = test_data["category"]

            print(f"\n🧪 Test {test_num}/{total_tests} ({category.upper()})")
            print(f"📝 ID: {prompt_id}")
            print(f"❓ Prompt: {original_prompt[:70]}...")

            self.logger.info(f"Running test {test_num}: {prompt_id} - {original_prompt[:50]}")

            # Generate enhanced prompt
            enhanced_prompt = self.concept_enhancer.enhance_prompt(original_prompt)

            print(f"⚡ Enhanced prompt length: {len(enhanced_prompt)} chars")

            # Generate responses
            print("🤖 Generating NORMAL response...")
            normal_response = self.generate_response(original_prompt, is_enhanced=False)

            print("🌟 Generating ICAR-ENHANCED response...")
            enhanced_response = self.generate_response(enhanced_prompt, is_enhanced=True)

            # Log responses
            self.logger.info(f"Normal response ({len(normal_response)} chars): {normal_response[:100]}")
            self.logger.info(f"Enhanced response ({len(enhanced_response)} chars): {enhanced_response[:100]}")

            # Evaluate
            evaluation = self.evaluator.evaluate_response_pair(test_data, normal_response, enhanced_response)
            improvement = evaluation.improvement_percentage

            print(f"📊 Normal response: {len(normal_response)} chars")
            print(f"📊 Enhanced response: {len(enhanced_response)} chars")
            print(f"🎯 IMPROVEMENT: {improvement:.2f}%")

            # Detailed scores
            scores = evaluation.scores
            print(f"📈 Relevance: {scores['relevance_normal']:.3f} → {scores['relevance_enhanced']:.3f}")
            print(f"📈 Concept: {scores['concept_normal']:.3f} → {scores['concept_enhanced']:.3f}")
            print(f"📈 Quality: {scores['quality_normal']:.3f} → {scores['quality_enhanced']:.3f}")

            self.logger.info(f"Test {test_num} completed. Improvement: {improvement:.2f}%")

            return {
                "test_number": test_num,
                "prompt_id": prompt_id,
                "category": category,
                "original_prompt": original_prompt,
                "enhanced_prompt": enhanced_prompt,
                "normal_response": normal_response,
                "enhanced_response": enhanced_response,
                "improvement_percentage": improvement,
                "detailed_scores": scores,
                "normal_length": len(normal_response),
                "enhanced_length": len(enhanced_response),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"❌ Error in test {test_num}: {str(e)}")
            self.logger.error(f"Error in test {test_num}: {str(e)}")
            return None

    def run_massive_test(self, limit: int = 1000) -> Dict:
        """Run 1000 prompt test"""
        print("🚀 STARTING MASSIVE TINYLLAMA TEST")
        print("=" * 60)
        print(f"🎯 Target: {limit} prompts")
        print(f"🧠 Model: TinyLlama 1.1B (1.1 billion parameters)")
        print(f"🔬 Comparison: ICAR vs NO-ICAR")
        print("=" * 60)

        # Load dataset
        test_dataset = self.load_test_dataset()
        if not test_dataset:
            return {}

        # Limit tests
        test_dataset = test_dataset[:limit]
        total_tests = len(test_dataset)

        # Load model
        if not self.load_model():
            return {}

        # Run tests
        results = []
        successful_tests = 0
        total_improvement = 0
        category_stats = {}

        start_time = time.time()

        for i, test_data in enumerate(test_dataset, 1):
            result = self.run_single_test(test_data, i, total_tests)

            if result:
                results.append(result)
                successful_tests += 1
                total_improvement += result["improvement_percentage"]

                # Category stats
                category = result["category"]
                if category not in category_stats:
                    category_stats[category] = {"count": 0, "total_improvement": 0, "improvements": []}

                category_stats[category]["count"] += 1
                category_stats[category]["total_improvement"] += result["improvement_percentage"]
                category_stats[category]["improvements"].append(result["improvement_percentage"])

                # Progress update
                if i % 50 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / i
                    remaining = (total_tests - i) * avg_time
                    print(f"\n⏰ Progress: {i}/{total_tests} ({(i/total_tests)*100:.1f}%)")
                    print(f"⏱️  Estimated remaining: {remaining/60:.1f} minutes")
                    print(f"📈 Current average improvement: {total_improvement/successful_tests:.2f}%")

            # Save intermediate results every 100 tests
            if i % 100 == 0:
                self.save_intermediate_results(results, i)

            # Small delay to prevent overheating
            time.sleep(0.5)

        # Calculate final results
        avg_improvement = total_improvement / successful_tests if successful_tests > 0 else 0

        # Category analysis
        for category, stats in category_stats.items():
            stats["avg_improvement"] = stats["total_improvement"] / stats["count"]
            stats["positive_count"] = len([x for x in stats["improvements"] if x > 0])
            stats["success_rate"] = (stats["positive_count"] / stats["count"]) * 100

        total_time = time.time() - start_time

        final_results = {
            "test_info": {
                "model": self.model_name,
                "total_prompts": total_tests,
                "successful_tests": successful_tests,
                "total_time_minutes": total_time / 60,
                "avg_time_per_test": total_time / successful_tests if successful_tests > 0 else 0
            },
            "overall_results": {
                "average_improvement": avg_improvement,
                "positive_improvements": len([r for r in results if r["improvement_percentage"] > 0]),
                "success_rate": (len([r for r in results if r["improvement_percentage"] > 0]) / successful_tests) * 100 if successful_tests > 0 else 0
            },
            "category_analysis": category_stats,
            "detailed_results": results
        }

        # Save final results
        self.save_final_results(final_results)

        # Print summary
        self.print_final_summary(final_results)

        return final_results

    def save_intermediate_results(self, results: List[Dict], test_count: int):
        """Save intermediate results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"intermediate_results_{test_count}_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 Intermediate results saved: {filename}")
        self.logger.info(f"Saved intermediate results: {filename}")

    def save_final_results(self, results: Dict):
        """Save final comprehensive results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"FINAL_MASSIVE_TEST_RESULTS_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 FINAL RESULTS SAVED: {filename}")
        self.logger.info(f"Final results saved: {filename}")

    def print_final_summary(self, results: Dict):
        """Print comprehensive final summary"""
        print("\n" + "="*80)
        print("🏆 MASSIVE TINYLLAMA TEST - FINAL RESULTS")
        print("="*80)

        test_info = results["test_info"]
        overall = results["overall_results"]
        categories = results["category_analysis"]

        print(f"🤖 Model: {test_info['model']}")
        print(f"📊 Total Tests: {test_info['successful_tests']}/{test_info['total_prompts']}")
        print(f"⏱️  Total Time: {test_info['total_time_minutes']:.1f} minutes")
        print(f"⚡ Avg Time per Test: {test_info['avg_time_per_test']:.1f} seconds")

        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"📈 Average Improvement: {overall['average_improvement']:.2f}%")
        print(f"✅ Positive Improvements: {overall['positive_improvements']}")
        print(f"🎯 Success Rate: {overall['success_rate']:.1f}%")

        print(f"\n📊 CATEGORY BREAKDOWN:")
        for category, stats in categories.items():
            print(f"  {category.upper()}:")
            print(f"    Tests: {stats['count']}")
            print(f"    Avg Improvement: {stats['avg_improvement']:.2f}%")
            print(f"    Success Rate: {stats['success_rate']:.1f}%")

        if overall['average_improvement'] > 20:
            print("\n🎉 EXCELLENT! ICAR shows STRONG improvement with TinyLlama!")
        elif overall['average_improvement'] > 10:
            print("\n👍 GOOD! ICAR shows positive improvement!")
        elif overall['average_improvement'] > 0:
            print("\n✅ POSITIVE! ICAR enhancement is working!")
        else:
            print("\n🤔 Mixed results, but valuable data obtained!")

        print("="*80)

def main():
    """Main execution"""
    tester = MassiveTinyLlamaTest()
    results = tester.run_massive_test(limit=1000)
    return results

if __name__ == "__main__":
    print("🚀 MASSIVE TINYLLAMA ICAR TEST - 1000 PROMPTS")
    print("=" * 60)
    results = main()
    print("🏁 Test completed!")