"""
Test with larger but CPU-friendly models
"""

import json
from pathlib import Path
from datetime import datetime
from transformers import pipeline
import torch

from concept_enhancer import PromptConceptEnhancer
from test_cases import TestCasesDataset
from evaluation_metrics import ResponseEvaluator

def test_models():
    """Test different model sizes"""

    models_to_test = [
        {
            "name": "gpt2",
            "size": "124M parameters",
            "description": "GPT-2 small - general purpose"
        },
        {
            "name": "gpt2-medium",
            "size": "355M parameters",
            "description": "GPT-2 medium - better quality"
        },
        {
            "name": "distilgpt2",
            "size": "82M parameters",
            "description": "Distilled GPT-2 - faster"
        }
    ]

    concept_enhancer = PromptConceptEnhancer()
    test_dataset = TestCasesDataset()
    evaluator = ResponseEvaluator()

    # Test 5 cases with each model
    test_cases = test_dataset.get_test_cases()[:5]

    results = {}

    for model_info in models_to_test:
        model_name = model_info["name"]
        print(f"\n🧪 Testing {model_name} ({model_info['size']})")
        print(f"📝 {model_info['description']}")

        try:
            # Load model
            generator = pipeline(
                "text-generation",
                model=model_name,
                max_length=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256,
                device=-1  # CPU
            )

            model_results = []
            total_improvement = 0

            for i, test_case in enumerate(test_cases, 1):
                print(f"  🔍 Test {i}: {test_case['query'][:40]}...")

                # Get responses
                original_query = test_case["query"]
                enhanced_prompt = concept_enhancer.enhance_prompt(original_query)

                # Generate responses
                try:
                    normal_result = generator(original_query, max_new_tokens=50)
                    enhanced_result = generator(enhanced_prompt, max_new_tokens=50)

                    normal_response = normal_result[0]['generated_text'][len(original_query):].strip()
                    enhanced_response = enhanced_result[0]['generated_text'][len(enhanced_prompt):].strip()

                    if not normal_response:
                        normal_response = "No response generated"
                    if not enhanced_response:
                        enhanced_response = "No response generated"

                    # Evaluate
                    result = evaluator.evaluate_response_pair(test_case, normal_response, enhanced_response)
                    improvement = result.improvement_percentage
                    total_improvement += improvement

                    print(f"    📊 Improvement: {improvement:.1f}%")

                    model_results.append({
                        "test": test_case['query'],
                        "improvement": improvement,
                        "normal_length": len(normal_response),
                        "enhanced_length": len(enhanced_response)
                    })

                except Exception as e:
                    print(f"    ❌ Error: {str(e)[:50]}")
                    continue

            avg_improvement = total_improvement / len(model_results) if model_results else 0

            results[model_name] = {
                "info": model_info,
                "avg_improvement": avg_improvement,
                "tests": model_results,
                "success_count": len(model_results)
            }

            print(f"  🎯 Average improvement: {avg_improvement:.1f}%")

        except Exception as e:
            print(f"  ❌ Failed to load {model_name}: {str(e)[:100]}")
            results[model_name] = {"error": str(e), "info": model_info}

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path("results") / f"model_comparison_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved: {results_file}")

    # Summary
    print(f"\n🏆 MODEL COMPARISON SUMMARY:")
    for model_name, data in results.items():
        if "error" not in data:
            info = data["info"]
            avg = data["avg_improvement"]
            print(f"📊 {model_name} ({info['size']}): {avg:.1f}% improvement")
        else:
            print(f"❌ {model_name}: Failed to load")

if __name__ == "__main__":
    print("🚀 ICAR MODEL COMPARISON TEST")
    print("=" * 50)
    test_models()