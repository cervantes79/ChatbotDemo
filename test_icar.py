#!/usr/bin/env python3
"""
Comprehensive ICAR Test Suite
Tests the Intelligent Concept-Aware RAG system
Author: Barış Genç
"""

import os
import sys
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append('src')

class TestConceptExtractor(unittest.TestCase):
    """Test the concept extraction functionality"""
    
    def setUp(self):
        from concept_extractor import ConceptExtractor
        self.extractor = ConceptExtractor()
    
    def test_document_concept_extraction(self):
        """Test extracting concepts from documents"""
        test_text = """
        This is a company policy document about employee benefits.
        The company provides health insurance, vacation days, and retirement plans.
        All employees must follow the dress code policy during work hours.
        """
        
        concepts = self.extractor.extract_document_concepts(test_text, "test_doc")
        
        self.assertGreater(len(concepts), 0)
        concept_names = [c.name for c in concepts]
        self.assertIn("business", concept_names)
        
    def test_query_concept_extraction(self):
        """Test extracting concepts from queries"""
        # Test greeting
        greeting_concepts = self.extractor.extract_query_concepts("Hello, how are you?")
        greeting_names = [c.name for c in greeting_concepts]
        self.assertIn("greeting", greeting_names)
        
        # Test weather query
        weather_concepts = self.extractor.extract_query_concepts("What's the weather in London?")
        weather_names = [c.name for c in weather_concepts]
        self.assertIn("weather_request", weather_names)
    
    def test_concept_matching(self):
        """Test concept matching between query and documents"""
        # Add some document concepts
        doc_text = "This document discusses employee policies and company procedures."
        self.extractor.extract_document_concepts(doc_text, "policy_doc")
        
        # Test query matching
        query_concepts = self.extractor.extract_query_concepts("What are the employee benefits?")
        matches = self.extractor.match_concepts(query_concepts)
        
        self.assertIsInstance(matches, list)

class TestGenericProcessor(unittest.TestCase):
    """Test the generic NLP processor"""
    
    def setUp(self):
        from generic_processor import GenericProcessor
        self.processor = GenericProcessor(processing_mode="hybrid")
    
    def test_keyword_extraction(self):
        """Test TF-IDF keyword extraction"""
        test_text = """
        Machine learning is a subset of artificial intelligence that focuses on algorithms.
        These algorithms can learn from data without being explicitly programmed.
        """
        
        keywords = self.processor.extract_keywords_tfidf(test_text, top_k=5)
        self.assertIsInstance(keywords, list)
        self.assertGreater(len(keywords), 0)
    
    def test_extractive_summarization(self):
        """Test extractive summarization"""
        test_text = """
        The quick brown fox jumps over the lazy dog. This is the first sentence.
        The second sentence explains more details about the fox's behavior.
        Finally, the third sentence concludes the story with the dog's reaction.
        """
        
        summary = self.processor.extractive_summarization(test_text, num_sentences=2)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertLess(len(summary), len(test_text))
    
    def test_document_chunking(self):
        """Test document chunking"""
        long_text = "This is a test sentence. " * 100  # Create long text
        
        chunks = self.processor.chunk_document(long_text, "test_doc")
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)
    
    def test_chunk_processing(self):
        """Test processing individual chunks"""
        chunk_text = "This is a test chunk about machine learning and data science."
        
        processed = self.processor.process_chunk(chunk_text, "test_doc", 0)
        
        self.assertEqual(processed.doc_id, "test_doc")
        self.assertEqual(processed.position, 0)
        self.assertIsInstance(processed.keywords, list)
        self.assertIsInstance(processed.summary, str)

class TestICARChatbot(unittest.TestCase):
    """Test the main ICAR chatbot functionality"""
    
    def setUp(self):
        from agent_core import ICARChatbot
        self.chatbot = ICARChatbot()
        
        # Mock the LLM to avoid API calls
        self.chatbot.llm = MagicMock()
        self.chatbot.llm.invoke.return_value = MagicMock(content="Mocked response")
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_initialization(self):
        """Test chatbot initialization"""
        with patch('src.vector_store.VectorStore.initialize', return_value=True):
            result = self.chatbot.initialize()
            self.assertTrue(result)
            self.assertTrue(self.chatbot.is_initialized)
    
    def test_decision_making(self):
        """Test ICAR decision making process"""
        # Test greeting detection
        action, reasoning = self.chatbot.decide_action("Hello there!")
        self.assertEqual(action.value, "icar_direct_response")
        self.assertIn("greeting", reasoning.lower())
        
        # Test weather detection
        action, reasoning = self.chatbot.decide_action("What's the weather in Tokyo?")
        self.assertEqual(action.value, "icar_weather_api")
        self.assertIn("weather", reasoning.lower())
        
        # Test document query
        action, reasoning = self.chatbot.decide_action("Tell me about company policies")
        self.assertIn("icar", action.value)
    
    def test_query_processing(self):
        """Test complete query processing"""
        with patch.object(self.chatbot, 'is_initialized', True):
            result = self.chatbot.process_query("Hello!")
            
            self.assertIn('response', result)
            self.assertIn('reasoning', result)
            self.assertIn('action_taken', result)
            self.assertIn('ICAR', result['action_taken'])

class TestEnhancedVectorStore(unittest.TestCase):
    """Test the enhanced vector store with document reconstruction"""
    
    def setUp(self):
        from enhanced_vector_store import EnhancedVectorStore
        # Use temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = EnhancedVectorStore(persist_directory=self.temp_dir)
    
    def tearDown(self):
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('chromadb.PersistentClient')
    def test_initialization(self, mock_client):
        """Test vector store initialization"""
        mock_collection = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.get_collection.side_effect = ValueError("Collection not found")
        mock_client_instance.create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        result = self.vector_store.initialize()
        self.assertTrue(result)
    
    def test_document_store_operations(self):
        """Test document store save/load operations"""
        # Test saving
        self.vector_store.document_store = {
            "documents": {"test_doc": {"text": "test", "source": "test.txt", "metadata": {}}},
            "chunks": {}
        }
        self.vector_store._save_document_store()
        
        # Test loading
        self.vector_store.document_store = {"documents": {}, "chunks": {}}
        self.vector_store._load_document_store()
        
        self.assertIn("test_doc", self.vector_store.document_store["documents"])

class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete ICAR system"""
    
    def test_complete_workflow(self):
        """Test complete ICAR workflow without external dependencies"""
        from concept_extractor import ConceptExtractor
        from generic_processor import GenericProcessor
        
        # Test concept extraction -> processing pipeline
        extractor = ConceptExtractor()
        processor = GenericProcessor()
        
        test_document = """
        Company Policy Document
        
        Employee Benefits:
        - Health insurance coverage
        - Vacation and sick leave
        - Retirement savings plan
        
        Work Schedule:
        - Standard hours: 9 AM to 5 PM
        - Flexible work arrangements available
        - Remote work options
        """
        
        # Extract concepts
        concepts = extractor.extract_document_concepts(test_document, "policy_doc")
        self.assertGreater(len(concepts), 0)
        
        # Process document
        processed_chunks = processor.process_document(test_document, "policy_doc")
        self.assertGreater(len(processed_chunks), 0)
        
        # Test query processing
        query = "What are the work hours?"
        query_concepts = extractor.extract_query_concepts(query)
        self.assertGreater(len(query_concepts), 0)

def run_basic_tests():
    """Run basic functionality tests without heavy dependencies"""
    print("🧠 ICAR Test Suite - Basic Functionality")
    print("=" * 50)
    
    # Test concept extractor
    print("\n1. Testing Concept Extractor...")
    try:
        from concept_extractor import ConceptExtractor
        extractor = ConceptExtractor()
        
        # Test basic concept extraction
        concepts = extractor.extract_query_concepts("Hello, how are you?")
        concept_names = [c.name for c in concepts]
        
        if "greeting" in concept_names:
            print("✅ Greeting detection works")
        else:
            print("❌ Greeting detection failed")
            
        concepts = extractor.extract_query_concepts("What's the weather in London?")
        concept_names = [c.name for c in concepts]
        
        if "weather_request" in concept_names:
            print("✅ Weather query detection works")
        else:
            print("❌ Weather query detection failed")
            
    except Exception as e:
        print(f"❌ Concept Extractor error: {e}")
    
    # Test generic processor
    print("\n2. Testing Generic Processor...")
    try:
        from generic_processor import GenericProcessor
        processor = GenericProcessor()
        
        test_text = "This is a test document about machine learning and artificial intelligence."
        keywords = processor.extract_keywords_tfidf(test_text)
        
        if len(keywords) > 0:
            print("✅ Keyword extraction works")
        else:
            print("❌ Keyword extraction failed")
            
        summary = processor.extractive_summarization(test_text, num_sentences=1)
        
        if len(summary) > 0:
            print("✅ Summarization works")
        else:
            print("❌ Summarization failed")
            
    except Exception as e:
        print(f"❌ Generic Processor error: {e}")
    
    # Test ICAR agent
    print("\n3. Testing ICAR Agent...")
    try:
        from agent_core import ICARChatbot
        chatbot = ICARChatbot()
        
        # Test decision making
        action, reasoning = chatbot.decide_action("Hello!")
        if "icar_direct_response" in action.value:
            print("✅ ICAR decision making works")
        else:
            print("❌ ICAR decision making failed")
            
    except Exception as e:
        print(f"❌ ICAR Agent error: {e}")
    
    print("\n" + "=" * 50)
    print("Basic functionality test completed!")

def main():
    """Main test runner"""
    print("🧠 ICAR Comprehensive Test Suite")
    print("Author: Barış Genç")
    print("=" * 50)
    
    # Check if we should run full tests or basic tests
    if len(sys.argv) > 1 and sys.argv[1] == "--basic":
        run_basic_tests()
        return
    
    # Run full test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestConceptExtractor,
        TestGenericProcessor,
        TestICARChatbot,
        TestEnhancedVectorStore,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("🎉 All ICAR tests passed!")
        print("ICAR system is ready for deployment.")
    else:
        print("❌ Some tests failed.")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(main())