"""
Test ICAR with TinyLlama 1.1B - A real billion-parameter model
Author: Barış Genç
"""

import json
import time
from pathlib import Path
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

from concept_enhancer import PromptConceptEnhancer
from test_cases import TestCasesDataset
from evaluation_metrics import ResponseEvaluator

def test_tinyllama():
    """Test with TinyLlama 1.1B model"""

    print("🚀 TESTING ICAR WITH TINYLLAMA 1.1B")
    print("🧠 Model: 1.1 BILLION parameters!")
    print("=" * 50)

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    try:
        print("📥 Loading TinyLlama 1.1B model...")
        print("⏳ This may take a few minutes...")

        # Load model with chat format
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="auto" if torch.cuda.is_available() else None
        )

        # Create pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            device=0 if torch.cuda.is_available() else -1
        )

        print("✅ TinyLlama loaded successfully!")

        # Initialize components
        concept_enhancer = PromptConceptEnhancer()
        test_dataset = TestCasesDataset()
        evaluator = ResponseEvaluator()

        # Test with 10 cases
        test_cases = test_dataset.get_test_cases()[:10]
        results = []
        total_improvement = 0

        print(f"\n🧪 Running {len(test_cases)} tests")
        print("=" * 40)

        for i, test_case in enumerate(test_cases, 1):
            query = test_case["query"]
            print(f"\n📝 Test {i}/10: {query[:50]}...")

            try:
                # Prepare prompts for chat format
                original_prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{query}\n<|assistant|>\n"

                enhanced_query = concept_enhancer.enhance_prompt(query)
                enhanced_prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{enhanced_query}\n<|assistant|>\n"

                # Generate responses
                print("  🤖 Generating normal response...")
                normal_result = generator(original_prompt, max_new_tokens=80)
                normal_response = normal_result[0]['generated_text']

                # Extract response after assistant tag
                if "<|assistant|>" in normal_response:
                    normal_response = normal_response.split("<|assistant|>")[-1].strip()

                print("  ⚡ Generating enhanced response...")
                enhanced_result = generator(enhanced_prompt, max_new_tokens=80)
                enhanced_response = enhanced_result[0]['generated_text']

                # Extract response after assistant tag
                if "<|assistant|>" in enhanced_response:
                    enhanced_response = enhanced_response.split("<|assistant|>")[-1].strip()

                # Clean up responses
                if not normal_response:
                    normal_response = "No response generated"
                if not enhanced_response:
                    enhanced_response = "No response generated"

                # Evaluate
                result = evaluator.evaluate_response_pair(test_case, normal_response, enhanced_response)
                improvement = result.improvement_percentage
                total_improvement += improvement

                print(f"  📊 Normal length: {len(normal_response)} chars")
                print(f"  📊 Enhanced length: {len(enhanced_response)} chars")
                print(f"  🎯 Improvement: {improvement:.1f}%")

                results.append({
                    "test_id": test_case["id"],
                    "query": query,
                    "category": test_case["category"],
                    "normal_response": normal_response,
                    "enhanced_response": enhanced_response,
                    "improvement": improvement,
                    "scores": result.scores
                })

                # Small delay
                time.sleep(1)

            except Exception as e:
                print(f"  ❌ Error in test {i}: {str(e)[:100]}")
                continue

        # Calculate results
        avg_improvement = total_improvement / len(results) if results else 0
        positive_improvements = len([r for r in results if r["improvement"] > 0])

        print(f"\n🏆 TINYLLAMA 1.1B RESULTS:")
        print(f"✅ Successful tests: {len(results)}/10")
        print(f"📈 Average improvement: {avg_improvement:.2f}%")
        print(f"🎯 Positive improvements: {positive_improvements}/{len(results)}")
        print(f"🧠 Model size: 1.1 BILLION parameters")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path("results") / f"tinyllama_benchmark_{timestamp}.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "model": "TinyLlama-1.1B-Chat-v1.0",
                "parameters": "1.1 billion",
                "timestamp": timestamp,
                "avg_improvement": avg_improvement,
                "positive_improvements": positive_improvements,
                "total_tests": len(results),
                "results": results
            }, f, indent=2, ensure_ascii=False)

        print(f"💾 Results saved: {results_file}")

        # Show best improvements
        best_results = sorted(results, key=lambda x: x["improvement"], reverse=True)[:3]
        print(f"\n🥇 TOP 3 IMPROVEMENTS:")
        for i, result in enumerate(best_results, 1):
            print(f"{i}. {result['improvement']:.1f}% - {result['query'][:40]}...")

        return avg_improvement

    except Exception as e:
        print(f"❌ Error loading TinyLlama: {str(e)}")
        return None

if __name__ == "__main__":
    result = test_tinyllama()

    if result is not None:
        if result > 20:
            print("\n🎉 EXCELLENT! ICAR shows strong improvement with 1B+ model!")
        elif result > 10:
            print("\n👍 GOOD! ICAR shows positive improvement!")
        elif result > 0:
            print("\n✅ POSITIVE! ICAR enhancement is working!")
        else:
            print("\n🤔 Mixed results, but still valuable data!")
    else:
        print("\n❌ Test failed to complete")