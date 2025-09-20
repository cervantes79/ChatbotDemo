"""
ICAR Validation Test - Check if concept extraction works correctly
Author: Barış Genç
"""

import json
from concept_enhancer import PromptConceptEnhancer

def test_icar_accuracy():
    """Test ICAR concept extraction with known examples"""

    enhancer = PromptConceptEnhancer()

    # Test cases with expected results
    test_cases = [
        {
            "id": "business_1",
            "prompt": "What are the company work hours?",
            "expected_category": "business",
            "expected_concepts": ["work", "company", "hours"]
        },
        {
            "id": "technical_1",
            "prompt": "How do I optimize database performance?",
            "expected_category": "technical",
            "expected_concepts": ["database", "performance", "optimize"]
        },
        {
            "id": "weather_1",
            "prompt": "What's the weather like in Istanbul?",
            "expected_category": "weather",
            "expected_concepts": ["weather", "Istanbul"]
        },
        {
            "id": "complex_1",
            "prompt": "How would demographic shifts impact cultural identity in the context of digital economies?",
            "expected_category": "complex",
            "expected_concepts": ["demographic", "cultural", "digital", "economic"]
        },
        {
            "id": "greeting_1",
            "prompt": "Hello, how are you today?",
            "expected_category": "greeting",
            "expected_concepts": ["hello", "greeting"]
        },
        {
            "id": "general_1",
            "prompt": "What is artificial intelligence?",
            "expected_category": "general",
            "expected_concepts": ["artificial intelligence", "technology"]
        }
    ]

    results = []

    print("🔍 ICAR CONCEPT EXTRACTION VALIDATION")
    print("=" * 60)

    for test in test_cases:
        prompt = test["prompt"]
        expected_cat = test["expected_category"]

        print(f"\n📝 Testing: {prompt}")
        print(f"🎯 Expected category: {expected_cat}")

        # Get concept analysis
        analysis = enhancer.get_concept_summary(prompt)
        actual_cat = analysis["primary_category"]
        concepts = [c["name"] for c in analysis["top_concepts"]]

        print(f"🤖 ICAR category: {actual_cat}")
        print(f"🧠 ICAR concepts: {concepts}")

        # Generate enhanced prompt
        enhanced = enhancer.enhance_prompt(prompt)

        # Check if correct
        is_correct = actual_cat == expected_cat
        status = "✅ CORRECT" if is_correct else "❌ WRONG"

        print(f"📊 Result: {status}")

        if not is_correct:
            print("🚨 ICAR FAILED - Wrong category!")

        results.append({
            "test_id": test["id"],
            "original_prompt": prompt,
            "enhanced_prompt": enhanced,
            "expected_category": expected_cat,
            "actual_category": actual_cat,
            "expected_concepts": test["expected_concepts"],
            "actual_concepts": concepts,
            "is_correct": is_correct,
            "concept_analysis": analysis
        })

        print("-" * 40)

    # Summary
    correct_count = sum(1 for r in results if r["is_correct"])
    total_count = len(results)
    accuracy = (correct_count / total_count) * 100

    print(f"\n📊 ICAR ACCURACY SUMMARY:")
    print(f"✅ Correct: {correct_count}/{total_count}")
    print(f"📈 Accuracy: {accuracy:.1f}%")

    if accuracy < 70:
        print("🚨 ICAR ACCURACY TOO LOW - NEEDS FIXING!")
    elif accuracy < 90:
        print("⚠️  ICAR ACCURACY MODERATE - COULD BE IMPROVED")
    else:
        print("🎉 ICAR ACCURACY GOOD!")

    # Save results
    with open("results2/icar_validation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_tests": total_count,
                "correct_predictions": correct_count,
                "accuracy_percentage": accuracy
            },
            "detailed_results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved: results2/icar_validation_results.json")

    return accuracy, results

if __name__ == "__main__":
    accuracy, results = test_icar_accuracy()

    if accuracy < 70:
        print("\n🔧 RECOMMENDATION: Fix ICAR algorithm before running mass tests!")
    else:
        print("\n🚀 ICAR validated - ready for mass testing!")