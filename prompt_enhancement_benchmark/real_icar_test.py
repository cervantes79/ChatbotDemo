"""
GERÇEK ICAR TEST - NLTK Keyword Extraction
10 test case ile TinyLlama
Author: Barış Genç (düzeltilmiş)
"""

import nltk
import time
from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class RealICAREnhancer:
    """Gerçek ICAR - NLTK keyword extraction ile"""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Remove some words that might be important
        self.stop_words.discard('how')
        self.stop_words.discard('what')
        self.stop_words.discard('when')
        self.stop_words.discard('where')
        self.stop_words.discard('why')

    def extract_keywords(self, text: str) -> List[str]:
        """NLTK ile keyword extraction"""
        # Tokenize
        tokens = word_tokenize(text.lower())

        # Remove punctuation and stopwords
        keywords = []
        for token in tokens:
            if (token.isalpha() and
                token not in self.stop_words and
                len(token) > 2):
                # Lemmatize
                lemma = self.lemmatizer.lemmatize(token)
                keywords.append(lemma)

        return keywords

    def enhance_prompt(self, original_prompt: str) -> str:
        """ICAR enhancement - keywords only"""
        keywords = self.extract_keywords(original_prompt)

        if not keywords:
            return original_prompt

        # Keep max 5 most important keywords
        enhanced = ' '.join(keywords[:5])
        return enhanced

class TinyLlamaTest:
    """TinyLlama ile test"""

    def __init__(self):
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.generator = None
        self.icar = RealICAREnhancer()

    def load_model(self):
        """TinyLlama load"""
        print("🚀 Loading TinyLlama 1.1B...")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=80,
            temperature=0.7,
            do_sample=True,
            device=0 if torch.cuda.is_available() else -1
        )

        print("✅ TinyLlama loaded!")

    def generate_response(self, prompt: str) -> str:
        """Generate response"""
        try:
            chat_prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{prompt}\n<|assistant|>\n"

            result = self.generator(
                chat_prompt,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )

            generated_text = result[0]['generated_text']
            if "<|assistant|>" in generated_text:
                response = generated_text.split("<|assistant|>")[-1].strip()
            else:
                response = generated_text.strip()

            if len(response) > 300:
                response = response[:300] + "..."

            return response

        except Exception as e:
            return f"Error: {str(e)[:50]}"

    def run_test(self):
        """10 test case"""

        test_cases = [
            "What are the company work hours?",
            "How do I optimize database performance?",
            "What's the weather like in Istanbul?",
            "How can I access project management tools?",
            "What are the benefits of volunteering?",
            "How do I debug performance issues?",
            "What is cloud computing and how does it work?",
            "Who is responsible for customer support?",
            "How do I report security breach?",
            "What tools are best for deployment?"
        ]

        self.load_model()

        print("\n🧪 GERÇEK ICAR TEST - 10 CASE")
        print("=" * 50)

        results = []

        for i, original in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}/10")
            print(f"Original: {original}")

            # ICAR enhancement
            enhanced = self.icar.enhance_prompt(original)
            keywords = self.icar.extract_keywords(original)

            print(f"Keywords: {keywords}")
            print(f"Enhanced: {enhanced}")

            # Generate responses
            print("🤖 Generating normal response...")
            normal_response = self.generate_response(original)

            print("⚡ Generating ICAR response...")
            icar_response = self.generate_response(enhanced)

            print(f"Normal ({len(normal_response)} chars): {normal_response[:100]}...")
            print(f"ICAR ({len(icar_response)} chars): {icar_response[:100]}...")

            # Simple quality assessment
            normal_score = self.assess_quality(original, normal_response)
            icar_score = self.assess_quality(original, icar_response)

            improvement = ((icar_score - normal_score) / normal_score * 100) if normal_score > 0 else 0

            print(f"📊 Quality: Normal={normal_score:.2f}, ICAR={icar_score:.2f}")
            print(f"🎯 Improvement: {improvement:.1f}%")

            results.append({
                'test_id': i,
                'original': original,
                'enhanced': enhanced,
                'keywords': keywords,
                'normal_response': normal_response,
                'icar_response': icar_response,
                'normal_score': normal_score,
                'icar_score': icar_score,
                'improvement': improvement
            })

            print("-" * 40)
            time.sleep(1)

        # Summary
        avg_improvement = sum(r['improvement'] for r in results) / len(results)
        positive_count = len([r for r in results if r['improvement'] > 0])

        print(f"\n📊 SONUÇLAR:")
        print(f"✅ Ortalama improvement: {avg_improvement:.1f}%")
        print(f"👍 Pozitif sonuç: {positive_count}/10")
        print(f"🎯 Başarı oranı: {positive_count/10*100:.0f}%")

        return results

    def assess_quality(self, original: str, response: str) -> float:
        """Simple quality assessment"""
        if not response or len(response) < 10:
            return 0.1

        score = 0.5  # Base score

        # Length check
        if 20 <= len(response.split()) <= 100:
            score += 0.2

        # Keyword presence
        original_words = set(word.lower() for word in original.split())
        response_words = set(word.lower() for word in response.split())
        overlap = len(original_words.intersection(response_words))
        score += min(0.3, overlap * 0.1)

        return min(1.0, score)

if __name__ == "__main__":
    print("🚀 GERÇEK ICAR TEST BAŞLIYOR!")
    tester = TinyLlamaTest()
    results = tester.run_test()
    print("✅ Test tamamlandı!")