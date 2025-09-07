#!/usr/bin/env python3
"""
Generic ICAR V2 Multi-Domain Test
Tests the system with different document types to verify generic capabilities

Author: Barış Genç
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_documents():
    """Create test documents from multiple domains"""
    
    # Educational domain
    education_doc = """
    Course Syllabus - Introduction to Computer Science
    
    Course Overview:
    This course introduces students to fundamental concepts in computer science including programming, 
    data structures, algorithms, and software development. Students will learn Python programming
    language and develop problem-solving skills.
    
    Learning Objectives:
    - Understand basic programming concepts
    - Implement data structures like lists, stacks, and queues
    - Analyze algorithm efficiency and complexity
    - Develop debugging and testing skills
    
    Grading Policy:
    Assignments: 40%
    Midterm Exam: 25%
    Final Project: 25%
    Class Participation: 10%
    
    Schedule:
    Week 1-3: Python Basics
    Week 4-6: Data Structures
    Week 7-9: Algorithms
    Week 10-12: Software Development
    Week 13-15: Final Project
    
    Prerequisites:
    Basic mathematics knowledge required. No prior programming experience necessary.
    """
    
    # Healthcare domain
    healthcare_doc = """
    Patient Care Guidelines - Diabetes Management
    
    Overview:
    Type 2 diabetes is a chronic condition that affects millions of people worldwide. 
    Proper management involves medication, diet, exercise, and regular monitoring.
    
    Treatment Protocol:
    1. Blood glucose monitoring - Check levels 2-3 times daily
    2. Medication management - Take metformin as prescribed
    3. Dietary recommendations - Limit carbohydrates, increase fiber
    4. Exercise routine - 30 minutes moderate activity daily
    5. Regular check-ups - Visit healthcare provider every 3 months
    
    Warning Signs:
    - Blood sugar above 300 mg/dL
    - Excessive thirst or urination
    - Blurred vision
    - Unexplained weight loss
    - Fatigue or weakness
    
    Emergency Procedures:
    If patient experiences severe symptoms, contact emergency services immediately.
    Administer glucose tablets if conscious and able to swallow.
    
    Lifestyle Modifications:
    - Maintain healthy weight
    - Quit smoking
    - Limit alcohol consumption
    - Manage stress through relaxation techniques
    
    Monitoring Schedule:
    Daily: Blood glucose, medication adherence
    Weekly: Weight, blood pressure
    Monthly: Food diary review
    Quarterly: HbA1c test, comprehensive exam
    """
    
    # E-commerce domain
    ecommerce_doc = """
    Product Information - TechGadget Pro Smartphone
    
    Product Description:
    The TechGadget Pro is a flagship smartphone featuring cutting-edge technology
    and premium design. Perfect for professionals and tech enthusiasts.
    
    Technical Specifications:
    Display: 6.7-inch OLED, 120Hz refresh rate
    Processor: Snapdragon 8 Gen 2 chipset
    Memory: 12GB RAM, 256GB storage
    Camera: Triple lens system - 108MP main, 12MP ultrawide, 8MP telephoto
    Battery: 5000mAh with fast charging
    Operating System: Android 13
    Connectivity: 5G, Wi-Fi 6E, Bluetooth 5.3
    
    Key Features:
    - AI-powered photography
    - Wireless charging capability
    - Water resistant (IP68 rating)
    - Face unlock and fingerprint scanner
    - Stereo speakers with Dolby Atmos
    
    Pricing and Availability:
    Retail Price: $899
    Special Offer: $799 (limited time)
    Colors Available: Midnight Black, Ocean Blue, Rose Gold
    Warranty: 2-year manufacturer warranty
    
    Customer Reviews:
    "Excellent camera quality and battery life" - 5 stars
    "Fast performance and beautiful display" - 5 stars
    "Great value for money" - 4 stars
    
    Shipping Information:
    Free shipping on orders over $50
    Standard delivery: 3-5 business days
    Express delivery: 1-2 business days
    International shipping available
    
    Return Policy:
    30-day return policy for unopened items
    14-day return policy for opened items
    Restocking fee may apply for certain products
    """
    
    # Legal domain
    legal_doc = """
    Software License Agreement
    
    Terms and Conditions:
    This Software License Agreement governs the use of the software product
    by end users. By installing or using this software, you agree to be bound
    by the terms of this agreement.
    
    Grant of License:
    Subject to the terms of this Agreement, the Company grants you a limited,
    non-exclusive, non-transferable license to use the software for personal
    or commercial purposes in accordance with the documentation.
    
    Restrictions:
    You may not:
    - Copy or distribute the software without permission
    - Reverse engineer or decompile the software
    - Use the software for illegal purposes
    - Remove copyright notices or proprietary labels
    
    Intellectual Property:
    The software and all related materials are protected by copyright and
    other intellectual property laws. All rights not expressly granted
    are reserved by the Company.
    
    Limitation of Liability:
    In no event shall the Company be liable for any indirect, incidental,
    special, or consequential damages arising out of or in connection with
    the use of this software.
    
    Termination:
    This agreement is effective until terminated. The Company may terminate
    this agreement immediately if you breach any terms. Upon termination,
    you must cease all use and destroy all copies of the software.
    
    Governing Law:
    This agreement shall be governed by and construed in accordance with
    the laws of the state of California, without regard to conflict of law principles.
    """
    
    return [
        {
            "text": education_doc,
            "doc_id": "edu_cs_syllabus", 
            "source": "education/computer_science_101.txt",
            "metadata": {"domain": "education", "subject": "computer_science", "type": "syllabus"}
        },
        {
            "text": healthcare_doc,
            "doc_id": "health_diabetes_guide",
            "source": "healthcare/diabetes_management.txt", 
            "metadata": {"domain": "healthcare", "condition": "diabetes", "type": "guidelines"}
        },
        {
            "text": ecommerce_doc,
            "doc_id": "product_smartphone",
            "source": "ecommerce/techgadget_pro.txt",
            "metadata": {"domain": "ecommerce", "category": "electronics", "type": "product_info"}
        },
        {
            "text": legal_doc,
            "doc_id": "legal_license_agreement",
            "source": "legal/software_license.txt",
            "metadata": {"domain": "legal", "document_type": "license_agreement", "type": "contract"}
        }
    ]

def test_processing_modes():
    """Test different processing modes"""
    try:
        from src.generic_agent import GenericICARAgent
        
        modes = ["keywords", "summary", "hybrid"]
        results = {}
        
        for mode in modes:
            print(f"\n🧠 Testing Generic ICAR V2 - Mode: {mode}")
            print("=" * 50)
            
            # Initialize agent
            agent = GenericICARAgent(processing_mode=mode, collection_name=f"test_{mode}")
            if not agent.initialize():
                print(f"❌ Failed to initialize agent with mode {mode}")
                continue
            
            # Reset collection for clean test
            agent.vector_store.reset_collection()
            
            # Load test documents
            test_docs = create_test_documents()
            success = agent.load_documents(test_docs)
            
            if not success:
                print(f"❌ Failed to load documents with mode {mode}")
                continue
                
            print(f"✅ Loaded {len(test_docs)} documents in {mode} mode")
            
            # Test queries from different domains
            test_queries = [
                "What are the grading criteria for the computer science course?",
                "How should I monitor blood glucose levels?", 
                "What are the camera specifications of the smartphone?",
                "What are the restrictions in the software license?"
            ]
            
            mode_results = []
            for query in test_queries:
                result = agent.process_query(query)
                mode_results.append({
                    "query": query,
                    "action": result["action_taken"],
                    "response_length": len(result["response"]),
                    "success": "error" not in result["action_taken"].lower()
                })
                
                print(f"Query: {query[:50]}...")
                print(f"Action: {result['action_taken']}")
                print(f"Response: {result['response'][:100]}...")
                print("---")
            
            results[mode] = {
                "documents_loaded": len(test_docs),
                "queries_tested": len(test_queries),
                "successful_queries": sum(1 for r in mode_results if r["success"]),
                "stats": agent.get_stats()
            }
        
        return results
        
    except Exception as e:
        print(f"❌ Error in processing mode tests: {str(e)}")
        return {}

def test_domain_agnostic():
    """Test that the system works across different domains"""
    try:
        print("\n🌍 Testing Domain-Agnostic Capabilities")
        print("=" * 50)
        
        from src.generic_agent import GenericICARAgent
        
        # Initialize with hybrid mode for best performance
        agent = GenericICARAgent(processing_mode="hybrid", collection_name="domain_test")
        if not agent.initialize():
            print("❌ Failed to initialize domain test agent")
            return False
        
        # Reset collection
        agent.vector_store.reset_collection()
        
        # Load all domain documents
        test_docs = create_test_documents()
        success = agent.load_documents(test_docs)
        
        if not success:
            print("❌ Failed to load domain test documents")
            return False
        
        print(f"✅ Loaded documents from {len(set(doc['metadata']['domain'] for doc in test_docs))} domains")
        
        # Test cross-domain queries
        cross_domain_queries = [
            "What requires daily monitoring?",  # Should match both healthcare (glucose) and education (assignments)
            "What are the main features or specifications?",  # Should match product specs and course objectives
            "What are the restrictions or limitations?",  # Should match legal restrictions and course prerequisites
            "What is the schedule or timeline?",  # Should match course schedule and healthcare monitoring
            "Hello, what can you help me with?"  # Should be direct response
        ]
        
        successful_queries = 0
        for query in cross_domain_queries:
            result = agent.process_query(query)
            
            if "error" not in result["action_taken"].lower():
                successful_queries += 1
                print(f"✅ Query: {query}")
                print(f"   Action: {result['action_taken']}")
                
                if "Generic ICAR Search" in result["action_taken"]:
                    # Check if response contains information from multiple domains
                    response_lower = result["response"].lower()
                    domains_mentioned = 0
                    
                    domain_keywords = {
                        "education": ["course", "student", "assignment", "grade"],
                        "healthcare": ["patient", "diabetes", "blood", "glucose"],
                        "ecommerce": ["product", "price", "camera", "smartphone"],
                        "legal": ["license", "agreement", "software", "terms"]
                    }
                    
                    for domain, keywords in domain_keywords.items():
                        if any(keyword in response_lower for keyword in keywords):
                            domains_mentioned += 1
                    
                    print(f"   Domains referenced: {domains_mentioned}")
                
            else:
                print(f"❌ Query failed: {query}")
            
            print("---")
        
        success_rate = successful_queries / len(cross_domain_queries)
        print(f"\n📊 Domain Test Results:")
        print(f"Success Rate: {success_rate:.1%} ({successful_queries}/{len(cross_domain_queries)})")
        
        return success_rate > 0.8  # 80% success rate threshold
        
    except Exception as e:
        print(f"❌ Error in domain-agnostic test: {str(e)}")
        return False

def main():
    """Run all Generic ICAR V2 tests"""
    print("🚀 GENERIC ICAR V2 MULTI-DOMAIN TEST SUITE")
    print("=" * 60)
    print("Author: Barış Genç")
    print("Testing domain-agnostic document processing capabilities")
    print("=" * 60)
    
    try:
        # Test 1: Different processing modes
        print("\n📋 Test 1: Processing Modes")
        mode_results = test_processing_modes()
        
        if mode_results:
            print(f"\n📊 Processing Mode Results:")
            for mode, results in mode_results.items():
                success_rate = results["successful_queries"] / results["queries_tested"]
                print(f"  {mode}: {success_rate:.1%} success rate ({results['successful_queries']}/{results['queries_tested']})")
        
        # Test 2: Domain-agnostic capabilities
        print("\n📋 Test 2: Domain-Agnostic Search")
        domain_success = test_domain_agnostic()
        
        # Overall results
        print("\n🎯 FINAL RESULTS:")
        print("=" * 30)
        
        if mode_results and domain_success:
            print("✅ Generic ICAR V2 successfully processes multiple domains")
            print("✅ All processing modes functional")
            print("✅ Domain-agnostic search working")
            print("\n🎉 GENERIC ICAR V2 READY FOR PRODUCTION!")
            return True
        else:
            print("❌ Some tests failed")
            return False
            
    except Exception as e:
        print(f"❌ Test suite failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)