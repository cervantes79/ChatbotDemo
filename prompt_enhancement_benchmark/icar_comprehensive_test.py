"""
ICAR Comprehensive Prompt Enhancement Test - 100 Prompts
Based on ICAR methodology with detailed logging and analysis
"""

import time
import json
import sys
import os
from typing import List, Dict
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import logging

# Add prompts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'prompts'))
from universal_enhancer import ICARPromptEnhancer

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results2/icar_test_detailed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ICARComprehensiveTest:
    """Comprehensive ICAR-based prompt enhancement test with 100 prompts"""

    def __init__(self):
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.generator = None
        self.icar_enhancer = ICARPromptEnhancer()

    def load_model(self):
        """Load TinyLlama model"""
        logger.info("🚀 Loading TinyLlama 1.1B...")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            trust_remote_code=True
        )
        model = model.to(device)

        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=80,  # Reduced for faster processing
            temperature=0.5,    # Lower temperature for more consistent results
            do_sample=True,
            device=device
        )

        logger.info("✅ TinyLlama loaded!")

    def generate_response(self, prompt: str, test_id: int) -> str:
        """Generate response with detailed logging"""
        try:
            chat_prompt = f"<|system|>\\nYou are a helpful assistant.\\n<|user|>\\n{prompt}\\n<|assistant|>\\n"

            start_time = time.time()
            result = self.generator(
                chat_prompt,
                max_new_tokens=80,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            generation_time = time.time() - start_time

            generated_text = result[0]['generated_text']
            if "<|assistant|>" in generated_text:
                response = generated_text.split("<|assistant|>")[-1].strip()
            else:
                response = generated_text.strip()

            if len(response) > 300:
                response = response[:300] + "..."

            logger.info(f"Test {test_id}: Generated response in {generation_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Test {test_id}: Generation error: {str(e)}")
            return f"Error: {str(e)[:50]}"

    def assess_quality(self, original: str, response: str) -> Dict[str, float]:
        """Enhanced quality assessment with detailed metrics"""
        if not response or len(response) < 10:
            return {"overall": 0.1, "structure": 0.0, "density": 0.0, "relevance": 0.0}

        # 1. Structure Quality (0.4 points)
        structure_score = 0.0

        # Check for organized content
        if any(marker in response for marker in ['•', '*', '-', '1.', '2.', '**']):
            structure_score += 0.15

        # Check for clear sections
        if response.count('\\n\\n') >= 1 or response.count('\\n') >= 2:
            structure_score += 0.1

        # Check for comparative/analytical language
        if any(word in response.lower() for word in ['vs', 'versus', 'difference', 'compared', 'while', 'whereas', 'however', 'moreover']):
            structure_score += 0.15

        # 2. Information Density (0.3 points)
        word_count = len(response.split())
        unique_concepts = len(set(word.lower() for word in response.split() if len(word) > 4))

        density_score = 0.0
        if word_count > 0:
            density = unique_concepts / word_count
            if density >= 0.4:
                density_score = 0.3
            elif density >= 0.3:
                density_score = 0.2
            elif density >= 0.2:
                density_score = 0.1

        # 3. Relevance and Completeness (0.3 points)
        original_words = set(word.lower() for word in original.split() if len(word) > 3)
        response_words = set(word.lower() for word in response.split())
        overlap = len(original_words.intersection(response_words))

        relevance_score = 0.0
        if overlap >= 3:
            relevance_score += 0.2
        elif overlap >= 2:
            relevance_score += 0.15
        elif overlap >= 1:
            relevance_score += 0.1

        # Completeness check
        if len(response.split('.')) >= 3:
            relevance_score += 0.1

        overall_score = min(1.0, structure_score + density_score + relevance_score)

        return {
            "overall": overall_score,
            "structure": structure_score,
            "density": density_score,
            "relevance": relevance_score
        }

    def get_100_test_prompts(self) -> List[Dict]:
        """Generate 100 diverse test prompts across categories"""
        prompts = []

        # Technical prompts (30)
        technical_base = [
            "What are the best practices for database optimization in enterprise environments?",
            "How do I configure continuous integration pipelines for microservices?",
            "What are the performance implications of containerization?",
            "How do I troubleshoot memory leaks in production systems?",
            "What are the security considerations for REST API design?",
            "How can I implement efficient caching strategies?",
            "What are the differences between SQL and NoSQL databases?",
            "How do I design scalable distributed systems?",
            "What are the principles of DevOps automation?",
            "How can I optimize cloud resource utilization?"
        ]

        for i, prompt in enumerate(technical_base):
            for variation in range(3):
                if variation == 0:
                    prompts.append({"prompt": prompt, "category": "technical", "complexity": "medium"})
                elif variation == 1:
                    prompts.append({"prompt": f"Explain {prompt.lower()}", "category": "technical", "complexity": "low"})
                else:
                    prompts.append({"prompt": f"What are the advanced aspects of {prompt.lower()[12:]}", "category": "technical", "complexity": "high"})

        # Business prompts (25)
        business_prompts = [
            "How can I access the employee portal for vacation requests?",
            "What are the requirements for budget allocation approval?",
            "How do I submit expense reports through the system?",
            "What is the process for performance review scheduling?",
            "How can I request additional training resources?",
            "What are the company policies for remote work?",
            "How do I update my emergency contact information?",
            "What are the procedures for equipment procurement?",
            "How can I schedule meeting rooms effectively?",
            "What is the workflow for project milestone reporting?",
            "How do I access customer relationship management tools?",
            "What are the guidelines for vendor management?",
            "How can I track project deliverables?",
            "What is the process for strategic planning sessions?",
            "How do I implement change management procedures?",
            "What are the best practices for stakeholder communication?",
            "How can I optimize team collaboration processes?",
            "What are the requirements for compliance reporting?",
            "How do I manage cross-functional project dependencies?",
            "What is the framework for decision-making processes?",
            "How can I establish effective feedback mechanisms?",
            "What are the protocols for crisis management?",
            "How do I develop leadership development programs?",
            "What is the methodology for market analysis?",
            "How can I create sustainable business growth strategies?"
        ]

        for prompt in business_prompts:
            prompts.append({"prompt": prompt, "category": "business", "complexity": "medium"})

        # General knowledge prompts (20)
        general_prompts = [
            "Can you explain quantum physics concepts?",
            "What is the difference between classical music and jazz?",
            "How can renewable energy address climate change?",
            "What are the historical causes of economic inflation?",
            "How do neural networks process information?",
            "What are the principles of sustainable agriculture?",
            "How does the human immune system work?",
            "What are the effects of globalization on culture?",
            "How do ecosystems maintain biological balance?",
            "What are the fundamentals of financial markets?",
            "How does artificial intelligence impact society?",
            "What are the theories of human psychology?",
            "How do governments implement fiscal policy?",
            "What are the mechanisms of genetic inheritance?",
            "How does international trade affect economies?",
            "What are the principles of architectural design?",
            "How do social movements create change?",
            "What are the foundations of democratic governance?",
            "How does technology influence education?",
            "What are the concepts of space exploration?"
        ]

        for prompt in general_prompts:
            prompts.append({"prompt": prompt, "category": "general", "complexity": "medium"})

        # Weather/Simple prompts (15)
        simple_prompts = [
            "What is the weather like in Istanbul today?",
            "What time is it in New York?",
            "How many days until Christmas?",
            "What is the current stock price of Apple?",
            "When is the next full moon?",
            "What is the population of Tokyo?",
            "How tall is Mount Everest?",
            "What is the speed of light?",
            "When was the Declaration of Independence signed?",
            "What is the chemical formula for water?",
            "How many continents are there?",
            "What is the largest ocean?",
            "When did World War II end?",
            "What is the capital of Australia?",
            "How many bones are in the human body?"
        ]

        for prompt in simple_prompts:
            prompts.append({"prompt": prompt, "category": "factual", "complexity": "low"})

        # Complex multi-domain prompts (10)
        complex_prompts = [
            "How do machine learning algorithms impact financial trading systems and what are the regulatory implications?",
            "What are the interdisciplinary approaches to solving urban sustainability challenges in smart cities?",
            "How do biotechnology advances in gene therapy intersect with ethical considerations and healthcare policy?",
            "What are the geopolitical implications of renewable energy transitions on international trade relationships?",
            "How do cybersecurity frameworks address privacy concerns in healthcare data management systems?",
            "What are the economic and social impacts of automation on labor markets and education systems?",
            "How do climate change mitigation strategies integrate with sustainable business development models?",
            "What are the technological and ethical challenges of artificial intelligence in judicial decision-making?",
            "How do cross-cultural communication strategies affect international business negotiations and partnerships?",
            "What are the interdisciplinary approaches to addressing mental health in remote work environments?"
        ]

        for prompt in complex_prompts:
            prompts.append({"prompt": prompt, "category": "complex", "complexity": "high"})

        return prompts[:100]  # Ensure exactly 100 prompts

    def run_comprehensive_test(self):
        """Run comprehensive 100-prompt ICAR test"""

        test_prompts = self.get_100_test_prompts()
        self.load_model()

        logger.info(f"\\n🌍 ICAR COMPREHENSIVE ENHANCEMENT TEST - 100 PROMPTS")
        logger.info("=" * 80)
        logger.info("📖 ICAR-Based Approach:")
        logger.info("✅ Concept-aware prompt classification")
        logger.info("✅ Multi-level enhancement strategies")
        logger.info("✅ Semantic preservation with context reconstruction")
        logger.info("✅ Temperature=0.5 for consistent results")
        logger.info("✅ TinyLlama 1.1B Model")
        logger.info("=" * 80)

        results = []
        category_stats = {}
        complexity_stats = {}

        for i, test_case in enumerate(test_prompts, 1):
            original = test_case["prompt"]
            category = test_case["category"]
            complexity = test_case["complexity"]

            logger.info(f"\\n📝 Test {i}/100")
            logger.info(f"Category: {category} | Complexity: {complexity}")
            logger.info(f"Original: {original}")

            # ICAR enhancement
            start_time = time.time()
            enhanced = self.icar_enhancer.enhance_prompt(original)
            enhancement_time = time.time() - start_time

            logger.info(f"Enhanced: {enhanced}")

            enhancement_applied = enhanced != original
            if enhancement_applied:
                char_reduction = len(original) - len(enhanced)
                logger.info(f"📊 ICAR enhancement: {len(original)} → {len(enhanced)} chars (-{char_reduction})")

                # Show ICAR analysis
                analysis = self.icar_enhancer.analyze_enhancement(original)
                prompt_type = max(analysis['sentence_type'].items(), key=lambda x: x[1])
                logger.info(f"🔍 ICAR prompt type: {prompt_type[0]} ({prompt_type[1]:.2f})")

                similarity = self.icar_enhancer.calculate_semantic_similarity(original, enhanced)
                logger.info(f"🎯 Semantic similarity: {similarity:.3f}")
            else:
                char_reduction = 0
                logger.info("❌ No enhancement applied")

            # Generate responses
            logger.info("🤖 Generating responses...")
            normal_response = self.generate_response(original, i)

            if enhancement_applied:
                logger.info("⚡ Generating ICAR-enhanced response...")
                icar_response = self.generate_response(enhanced, i)
            else:
                logger.info("⚡ Using same response (no enhancement applied)...")
                icar_response = normal_response

            # Quality assessment
            normal_quality = self.assess_quality(original, normal_response)
            icar_quality = self.assess_quality(original, icar_response)

            improvement = ((icar_quality["overall"] - normal_quality["overall"]) / normal_quality["overall"] * 100) if normal_quality["overall"] > 0 else 0

            logger.info(f"📊 Quality: Normal={normal_quality['overall']:.3f}, ICAR={icar_quality['overall']:.3f}")
            logger.info(f"🎯 Improvement: {improvement:.1f}%")

            # Category and complexity statistics
            if category not in category_stats:
                category_stats[category] = {'count': 0, 'total_improvement': 0, 'positive': 0}
            category_stats[category]['count'] += 1
            category_stats[category]['total_improvement'] += improvement
            if improvement > 0:
                category_stats[category]['positive'] += 1

            if complexity not in complexity_stats:
                complexity_stats[complexity] = {'count': 0, 'total_improvement': 0, 'positive': 0}
            complexity_stats[complexity]['count'] += 1
            complexity_stats[complexity]['total_improvement'] += improvement
            if improvement > 0:
                complexity_stats[complexity]['positive'] += 1

            results.append({
                'test_id': i,
                'category': category,
                'complexity': complexity,
                'original': original,
                'enhanced': enhanced,
                'normal_response': normal_response,
                'icar_response': icar_response,
                'normal_quality': normal_quality,
                'icar_quality': icar_quality,
                'improvement': improvement,
                'enhancement_applied': enhancement_applied,
                'char_reduction': char_reduction,
                'enhancement_time': enhancement_time,
                'semantic_similarity': self.icar_enhancer.calculate_semantic_similarity(original, enhanced) if enhancement_applied else 1.0
            })

            # Progress update every 10 tests
            if i % 10 == 0:
                current_avg = sum(r['improvement'] for r in results) / len(results)
                logger.info(f"\\n🔄 Progress: {i}/100 tests completed, Current avg improvement: {current_avg:.1f}%")

            time.sleep(0.1)  # Small delay for stability

        # Final comprehensive analysis
        self.generate_comprehensive_report(results, category_stats, complexity_stats)

        return results

    def generate_comprehensive_report(self, results: List[Dict], category_stats: Dict, complexity_stats: Dict):
        """Generate detailed comprehensive report"""

        # Overall statistics
        avg_improvement = sum(r['improvement'] for r in results) / len(results)
        positive_count = len([r for r in results if r['improvement'] > 0])
        enhancements_applied = len([r for r in results if r['enhancement_applied']])
        avg_enhancement_time = sum(r['enhancement_time'] for r in results) / len(results)
        avg_char_reduction = sum(r['char_reduction'] for r in results if r['char_reduction'] > 0)
        if enhancements_applied > 0:
            avg_char_reduction = avg_char_reduction / enhancements_applied
        avg_similarity = sum(r['semantic_similarity'] for r in results) / len(results)

        logger.info(f"\\n📊 COMPREHENSIVE ICAR RESULTS (100 Prompts):")
        logger.info(f"✅ Average improvement: {avg_improvement:.1f}%")
        logger.info(f"👍 Positive results: {positive_count}/100")
        logger.info(f"🎯 Success rate: {positive_count/100*100:.0f}%")
        logger.info(f"⚡ Enhancements applied: {enhancements_applied}/100")
        logger.info(f"📊 Average char reduction: {avg_char_reduction:.0f} chars")
        logger.info(f"🎯 Average semantic similarity: {avg_similarity:.3f}")
        logger.info(f"⏱️ Average enhancement time: {avg_enhancement_time:.3f}s")

        # Category breakdown
        logger.info(f"\\n📊 CATEGORY BREAKDOWN:")
        for category, stats in category_stats.items():
            avg_cat_improvement = stats['total_improvement'] / stats['count']
            success_rate = (stats['positive'] / stats['count']) * 100
            logger.info(f"   {category}: {avg_cat_improvement:.1f}% avg, {success_rate:.0f}% success ({stats['count']} tests)")

        # Complexity breakdown
        logger.info(f"\\n📊 COMPLEXITY BREAKDOWN:")
        for complexity, stats in complexity_stats.items():
            avg_comp_improvement = stats['total_improvement'] / stats['count']
            success_rate = (stats['positive'] / stats['count']) * 100
            logger.info(f"   {complexity}: {avg_comp_improvement:.1f}% avg, {success_rate:.0f}% success ({stats['count']} tests)")

        # Best and worst cases
        best = max(results, key=lambda x: x['improvement'])
        worst = min(results, key=lambda x: x['improvement'])

        logger.info(f"\\n🏆 BEST CASE: Test {best['test_id']} (+{best['improvement']:.1f}%)")
        logger.info(f"   Category: {best['category']} | Complexity: {best['complexity']}")
        logger.info(f"   Original: {best['original']}")
        if best['enhancement_applied']:
            logger.info(f"   Enhanced: {best['enhanced']}")

        logger.info(f"\\n💥 WORST CASE: Test {worst['test_id']} ({worst['improvement']:.1f}%)")
        logger.info(f"   Category: {worst['category']} | Complexity: {worst['complexity']}")
        logger.info(f"   Original: {worst['original']}")

        # Save detailed results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"results2/icar_comprehensive_100_test_{timestamp}.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'approach': 'icar_concept_aware',
                    'model': 'TinyLlama-1.1B',
                    'total_tests': len(results),
                    'average_improvement': avg_improvement,
                    'positive_results': positive_count,
                    'success_rate': positive_count/100*100,
                    'enhancements_applied': enhancements_applied,
                    'avg_char_reduction': avg_char_reduction,
                    'avg_semantic_similarity': avg_similarity,
                    'avg_enhancement_time': avg_enhancement_time,
                    'category_stats': category_stats,
                    'complexity_stats': complexity_stats
                },
                'test_results': results
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"\\n💾 Comprehensive results saved to: {results_file}")

        # Historical comparison
        logger.info(f"\\n🔬 HISTORICAL COMPARISON:")
        logger.info(f"📈 Previous Universal (10 prompts): -3.5% avg, 20% success")
        logger.info(f"📈 ICAR Comprehensive (100 prompts): {avg_improvement:.1f}% avg, {positive_count/100*100:.0f}% success")

if __name__ == "__main__":
    logger.info("🚀 ICAR COMPREHENSIVE TEST STARTING!")
    logger.info("📖 ICAR-Based Concept-Aware Prompt Enhancement")
    tester = ICARComprehensiveTest()
    results = tester.run_comprehensive_test()
    logger.info("✅ ICAR comprehensive test completed!")