"""
ICAR-Based Concept-Aware Prompt Enhancement
Implements ICAR methodology for prompt optimization:
- Concept extraction and multi-level matching
- Intelligent action selection for prompt types
- Context reconstruction with semantic preservation
Based on: "ICAR: Intelligent Concept-Aware Retrieval-Augmented Generation"
"""

import nltk
import numpy as np
from typing import List, Tuple, Dict, Optional
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

# Download required NLTK data
required_downloads = [
    'punkt', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
    'stopwords', 'universal_tagset'
]

for item in required_downloads:
    try:
        nltk.data.find(f'tokenizers/{item}')
    except LookupError:
        try:
            nltk.data.find(f'taggers/{item}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{item}')
            except LookupError:
                nltk.download(item)

class ICARPromptEnhancer:
    """ICAR-based concept-aware prompt enhancement using multi-level matching"""

    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()

        # ICAR prompt type classification patterns
        self.prompt_types = {
            'concept-based': ['what', 'how', 'define', 'explain', 'describe'],
            'semantic': ['similar', 'like', 'related', 'compare'],
            'complex': ['implications', 'impact', 'effects', 'consequences'],
            'exact': ['is', 'are', 'when', 'where'],
            'ambiguous': ['best', 'better', 'optimal', 'should']
        }

    def classify_prompt_type(self, text: str) -> Dict[str, float]:
        """ICAR-based prompt type classification using concept analysis"""
        text_lower = text.lower().strip()

        scores = {
            'concept-based': 0.0,
            'semantic': 0.0,
            'complex': 0.0,
            'exact': 0.0,
            'ambiguous': 0.0
        }

        # ICAR prompt type classification
        for prompt_type, keywords in self.prompt_types.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[prompt_type] += 0.3

        # Question structure detection (maps to concept-based)
        if text.endswith('?'):
            scores['concept-based'] += 0.4

        question_words = ['what', 'how', 'when', 'where', 'why', 'which', 'who']
        for word in question_words:
            if text_lower.startswith(word):
                scores['concept-based'] += 0.3
                break

        # Modal verbs (maps to ambiguous)
        modal_words = ['can', 'could', 'would', 'will', 'should']
        for word in modal_words:
            if text_lower.startswith(word):
                scores['ambiguous'] += 0.3
                break

        return scores

    def calculate_semantic_importance(self, text: str) -> Dict[str, float]:
        """Calculate semantic importance of each word using multiple factors"""
        tokens = word_tokenize(text)

        if len(tokens) == 0:
            return {}

        # Universal POS tags for language independence
        try:
            pos_tags = pos_tag(tokens, tagset='universal')
        except:
            pos_tags = pos_tag(tokens)

        word_scores = {}

        # 1. TF-IDF based importance
        tfidf_scores = {}
        try:
            if len(text.strip()) > 0:
                vectorizer = TfidfVectorizer(max_features=50, stop_words=None)
                tfidf_matrix = vectorizer.fit_transform([text])
                feature_names = vectorizer.get_feature_names_out()
                tfidf_vals = tfidf_matrix.toarray()[0]

                for word, score in zip(feature_names, tfidf_vals):
                    tfidf_scores[word.lower()] = score
        except:
            pass

        # 2. Universal POS-based scoring
        pos_weights = {
            'NOUN': 1.0, 'PROPN': 1.2, 'VERB': 0.8, 'ADJ': 0.7,
            'ADV': 0.4, 'DET': 0.1, 'ADP': 0.2, 'CONJ': 0.1,
            'PRON': 0.2, 'PRT': 0.2, 'NUM': 0.6, 'X': 0.3
        }

        for word, pos in pos_tags:
            word_lower = word.lower()

            if len(word) <= 2:  # Skip very short words
                continue

            # Base score from POS
            base_score = pos_weights.get(pos, 0.5)

            # TF-IDF boost
            tfidf_boost = tfidf_scores.get(word_lower, 0) * 2

            # Length factor (longer words often more important)
            length_factor = min(1.2, len(word) / 6.0)

            # Final score
            final_score = (base_score + tfidf_boost) * length_factor
            word_scores[word_lower] = final_score

        return word_scores

    def generate_candidates(self, text: str, sentence_type: Dict[str, float], word_scores: Dict[str, float]) -> List[str]:
        """Generate multiple enhancement candidates"""
        tokens = word_tokenize(text)
        candidates = []

        # Strategy 1: Conservative (preserve structure, remove redundancy)
        conservative = self._conservative_enhancement(tokens, sentence_type, word_scores)
        if conservative and conservative != text:
            candidates.append(conservative)

        # Strategy 2: Moderate (balance structure and compression)
        moderate = self._moderate_enhancement(tokens, sentence_type, word_scores)
        if moderate and moderate != text:
            candidates.append(moderate)

        # Strategy 3: Aggressive (maximum compression while preserving meaning)
        aggressive = self._aggressive_enhancement(tokens, sentence_type, word_scores)
        if aggressive and aggressive != text:
            candidates.append(aggressive)

        return candidates

    def _conservative_enhancement(self, tokens: List[str], sentence_type: Dict[str, float], word_scores: Dict[str, float]) -> str:
        """Conservative enhancement - preserve structure, minimal changes"""
        if not tokens:
            return ""

        kept_tokens = []

        # If it's a question, preserve question structure
        if sentence_type.get('question', 0) > 0.3:
            # Keep first 2-3 words for question structure
            for i, token in enumerate(tokens[:3]):
                if token.lower() in ['what', 'how', 'when', 'where', 'why', 'which', 'who', 'can', 'could', 'are', 'is', 'do', 'does']:
                    kept_tokens.append(token.lower())
                elif i < 2:  # Keep auxiliary words
                    kept_tokens.append(token.lower())

        # Add high-importance content words
        for token in tokens:
            token_lower = token.lower()
            score = word_scores.get(token_lower, 0)

            # Skip if already added or if it's a very low importance word
            if token_lower in [t.lower() for t in kept_tokens]:
                continue

            if score > 0.5 and token_lower not in self.stop_words:
                kept_tokens.append(token_lower)

        return ' '.join(kept_tokens) if kept_tokens else ""

    def _moderate_enhancement(self, tokens: List[str], sentence_type: Dict[str, float], word_scores: Dict[str, float]) -> str:
        """Moderate enhancement - balance compression and structure"""
        if not tokens:
            return ""

        kept_tokens = []

        # Preserve essential question words but simplify structure
        if sentence_type.get('question', 0) > 0.3:
            first_word = tokens[0].lower()
            if first_word in ['what', 'how', 'when', 'where', 'why', 'which', 'who']:
                kept_tokens.append(first_word)

                # Add essential auxiliary if needed
                if len(tokens) > 1 and tokens[1].lower() in ['are', 'is', 'can', 'do']:
                    kept_tokens.append(tokens[1].lower())

        # Add top scoring words
        scored_tokens = [(token, word_scores.get(token.lower(), 0)) for token in tokens]
        scored_tokens.sort(key=lambda x: x[1], reverse=True)

        for token, score in scored_tokens[:6]:  # Limit to top 6 content words
            token_lower = token.lower()
            if (token_lower not in [t.lower() for t in kept_tokens] and
                score > 0.4 and
                token_lower not in self.stop_words and
                len(token) > 2):
                kept_tokens.append(token_lower)

        return ' '.join(kept_tokens) if kept_tokens else ""

    def _aggressive_enhancement(self, tokens: List[str], sentence_type: Dict[str, float], word_scores: Dict[str, float]) -> str:
        """Aggressive enhancement - maximum compression"""
        if not tokens:
            return ""

        # Get only the highest scoring words
        scored_tokens = [(token, word_scores.get(token.lower(), 0)) for token in tokens]
        scored_tokens.sort(key=lambda x: x[1], reverse=True)

        kept_tokens = []
        for token, score in scored_tokens[:5]:  # Only top 5 words
            token_lower = token.lower()
            if score > 0.6 and len(token) > 2 and token_lower not in self.stop_words:
                kept_tokens.append(token_lower)

        return ' '.join(kept_tokens) if kept_tokens else ""

    def calculate_semantic_similarity(self, original: str, candidate: str) -> float:
        """Calculate semantic similarity between original and candidate"""
        if not original or not candidate:
            return 0.0

        try:
            vectorizer = TfidfVectorizer(stop_words=None)
            vectors = vectorizer.fit_transform([original, candidate])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return similarity
        except:
            # Fallback: simple word overlap
            original_words = set(original.lower().split())
            candidate_words = set(candidate.lower().split())
            if not original_words:
                return 0.0
            overlap = len(original_words.intersection(candidate_words))
            return overlap / len(original_words)

    def enhance_prompt(self, original_prompt: str) -> str:
        """Main enhancement function with semantic preservation"""
        if not original_prompt or len(original_prompt.strip()) < 5:
            return original_prompt

        # 1. Analyze sentence structure
        sentence_type = self.classify_prompt_type(original_prompt)

        # 2. Calculate word importance
        word_scores = self.calculate_semantic_importance(original_prompt)

        # 3. Generate candidates
        candidates = self.generate_candidates(original_prompt, sentence_type, word_scores)

        if not candidates:
            return original_prompt

        # 4. Select best candidate based on semantic similarity + compression ratio
        best_candidate = original_prompt
        best_score = 0.0

        for candidate in candidates:
            # Calculate semantic similarity
            similarity = self.calculate_semantic_similarity(original_prompt, candidate)

            # Calculate compression ratio
            compression = 1.0 - (len(candidate) / len(original_prompt))

            # Combined score (balance similarity and compression)
            combined_score = 0.7 * similarity + 0.3 * compression

            # Minimum similarity threshold to preserve meaning
            if similarity >= 0.6 and combined_score > best_score:
                best_score = combined_score
                best_candidate = candidate

        # Only return enhanced version if it's meaningfully different
        if best_candidate != original_prompt and len(best_candidate.split()) >= 3:
            word_reduction = len(original_prompt.split()) - len(best_candidate.split())
            if word_reduction >= 2:
                return best_candidate

        return original_prompt

    def analyze_enhancement(self, original: str) -> Dict:
        """Analyze enhancement process for debugging"""
        sentence_type = self.classify_prompt_type(original)
        word_scores = self.calculate_semantic_importance(original)
        candidates = self.generate_candidates(original, sentence_type, word_scores)
        enhanced = self.enhance_prompt(original)

        return {
            'original': original,
            'enhanced': enhanced,
            'sentence_type': sentence_type,
            'top_words': dict(sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:8]),
            'candidates': candidates,
            'applied': enhanced != original
        }

def test_universal_enhancer():
    """Test the universal enhancer"""
    enhancer = UniversalPromptEnhancer()

    test_cases = [
        "What are the best practices for database optimization in enterprise environments?",
        "How do I configure continuous integration pipelines for microservices?",
        "What is the weather like in Istanbul today?",
        "Can you explain quantum physics concepts?",
        "How can I access the employee portal for vacation requests?"
    ]

    print("🌍 UNIVERSAL ICAR ENHANCEMENT TEST")
    print("=" * 60)

    for i, prompt in enumerate(test_cases, 1):
        analysis = enhancer.analyze_enhancement(prompt)

        print(f"\n📝 Test {i}")
        print(f"Original: {analysis['original']}")
        print(f"Enhanced: {analysis['enhanced']}")
        print(f"Applied: {analysis['applied']}")

        sentence_type = max(analysis['sentence_type'].items(), key=lambda x: x[1])
        print(f"Type: {sentence_type[0]} ({sentence_type[1]:.2f})")

        if analysis['candidates']:
            print(f"Candidates: {len(analysis['candidates'])}")
        print(f"Top words: {list(analysis['top_words'].keys())[:5]}")

        if analysis['applied']:
            similarity = enhancer.calculate_semantic_similarity(analysis['original'], analysis['enhanced'])
            print(f"Similarity: {similarity:.3f}")

        print("-" * 40)

if __name__ == "__main__":
    test_universal_enhancer()