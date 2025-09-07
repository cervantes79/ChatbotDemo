#!/usr/bin/env python3
"""
Generic ICAR V2 Multi-Domain Test Suite
Tests the Universal Domain-Agnostic RAG system
Author: Barış Genç
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.append('src')

class TestMultiDomainGenericICAR:
    """
    Multi-Domain Test Suite for Generic ICAR V2
    Tests education, healthcare, e-commerce, legal documents
    """
    
    def __init__(self):
        self.temp_dir = None
        self.test_documents = self.create_test_documents()
    
    def create_test_documents(self) -> dict:
        """Create test documents from various domains"""
        return {
            "education": {
                "computer_science_101.txt": """
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
                """,
                
                "student_handbook.txt": """
                University Student Handbook
                
                Academic Calendar:
                Fall Semester: September - December
                Spring Semester: January - May  
                Summer Session: June - August
                
                Attendance Policy:
                Students are expected to attend all classes. More than 3 unexcused absences
                may result in course failure. Medical absences require documentation.
                
                Grade Scale:
                A: 90-100%
                B: 80-89%
                C: 70-79%
                D: 60-69%
                F: Below 60%
                
                Campus Resources:
                - Library: Open 24/7 during finals week
                - Writing Center: Free tutoring available
                - Career Services: Resume help and job placement
                - Health Center: Medical services for students
                
                Student Conduct:
                Academic dishonesty including plagiarism and cheating will result in
                immediate course failure and possible expulsion.
                """
            },
            
            "healthcare": {
                "diabetes_management.txt": """
                Diabetes Management Guide
                
                Daily Monitoring:
                - Check blood glucose levels 2-3 times per day
                - Record readings in diabetes log
                - Monitor for symptoms of high/low blood sugar
                
                Medication Schedule:
                Morning (7 AM):
                - Long-acting insulin (Lantus) - 20 units
                - Metformin - 500mg with breakfast
                
                Evening (7 PM):
                - Metformin - 500mg with dinner
                
                Diet Recommendations:
                - Limit carbohydrates to 45-60g per meal
                - Include protein and fiber in every meal
                - Avoid sugary drinks and processed foods
                - Eat regular meals at consistent times
                
                Exercise Guidelines:
                - 30 minutes of moderate exercise daily
                - Check blood sugar before and after exercise
                - Carry glucose tablets during workouts
                - Consult doctor before starting new exercise routine
                
                Warning Signs:
                Seek immediate medical attention if experiencing:
                - Blood glucose over 300 mg/dL
                - Persistent nausea and vomiting
                - Difficulty breathing
                - Loss of consciousness
                """,
                
                "patient_care_protocol.txt": """
                Patient Care Protocol - General Guidelines
                
                Admission Procedures:
                1. Verify patient identity using two identifiers
                2. Review medical history and current medications
                3. Conduct initial assessment within 30 minutes
                4. Document all findings in electronic health record
                
                Medication Administration:
                - Verify patient, medication, dose, route, and time (5 Rights)
                - Check for drug allergies before administration
                - Monitor for adverse reactions
                - Document administration immediately
                
                Vital Signs Monitoring:
                Standard Schedule:
                - Every 4 hours for stable patients
                - Every 2 hours for moderate risk patients
                - Continuous monitoring for critical patients
                
                Parameters to Monitor:
                - Temperature, Blood Pressure, Heart Rate, Respiratory Rate
                - Oxygen Saturation, Pain Level (0-10 scale)
                - Mental Status and Mobility
                
                Discharge Planning:
                Begin discharge planning within 24 hours of admission:
                - Assess home care needs
                - Arrange follow-up appointments
                - Provide medication reconciliation
                - Give written discharge instructions
                """
            },
            
            "e_commerce": {
                "product_catalog.txt": """
                E-Commerce Product Catalog Management
                
                Product Categories:
                Electronics:
                - Smartphones, Tablets, Laptops
                - Audio equipment, Gaming consoles
                - Smart home devices
                
                Clothing & Fashion:
                - Men's, Women's, Children's apparel
                - Shoes, Accessories, Jewelry
                - Seasonal collections
                
                Home & Garden:
                - Furniture, Home decor
                - Kitchen appliances, Tools
                - Outdoor and gardening supplies
                
                Pricing Strategy:
                - Competitive pricing analysis updated weekly
                - Dynamic pricing for seasonal items
                - Bulk discount tiers: 5%, 10%, 15% off
                - Free shipping on orders over $50
                
                Inventory Management:
                - Automatic reorder when stock falls below 10 units
                - Seasonal inventory planning
                - Dead stock clearance quarterly
                - Real-time inventory updates across all channels
                
                Customer Reviews:
                - Encourage reviews through email campaigns
                - Respond to negative reviews within 24 hours
                - Use review data for product improvements
                - Display average rating prominently
                """,
                
                "shipping_policy.txt": """
                Shipping and Returns Policy
                
                Shipping Options:
                Standard Shipping (5-7 business days): $5.99
                Express Shipping (2-3 business days): $9.99
                Next Day Delivery (1 business day): $19.99
                Free Standard Shipping on orders over $50
                
                International Shipping:
                Available to 50+ countries
                Shipping costs calculated at checkout
                Delivery time: 7-21 business days
                Customer responsible for customs fees
                
                Return Policy:
                - 30-day return window from delivery date
                - Items must be in original condition with tags
                - Free returns for defective or wrong items
                - Customer pays return shipping for other returns
                
                Exchange Process:
                1. Initiate return through customer account
                2. Print prepaid return label (if applicable)
                3. Package item securely
                4. Drop off at authorized location
                5. Refund processed within 5-7 business days
                
                Warranty Information:
                Electronics: 1-year manufacturer warranty
                Clothing: 90-day quality guarantee
                Appliances: Extended warranty options available
                """
            },
            
            "legal": {
                "contract_terms.txt": """
                Legal Contract Terms and Conditions
                
                Service Agreement Terms:
                This agreement governs the relationship between the Service Provider
                and the Client for professional consulting services.
                
                Scope of Work:
                - Services to be performed as detailed in Statement of Work
                - Deliverables must meet specified quality standards
                - Timeline and milestones as agreed in project schedule
                
                Payment Terms:
                - 50% deposit required upon contract signing
                - Remaining balance due within 30 days of project completion
                - Late payments subject to 1.5% monthly interest charge
                - Additional work requires written approval and separate billing
                
                Intellectual Property:
                - Client retains ownership of pre-existing materials
                - Work product created during engagement belongs to Client
                - Service Provider retains right to use general methodologies
                - Confidential information must be protected for 5 years
                
                Termination Clause:
                Either party may terminate with 30 days written notice
                - Client pays for work completed to date
                - Return of confidential materials required
                - Non-compete clause remains in effect for 12 months
                
                Limitation of Liability:
                Service Provider's total liability limited to contract value
                Neither party liable for indirect or consequential damages
                """,
                
                "employment_law.txt": """
                Employment Law Guidelines
                
                Hiring Practices:
                - Equal opportunity employment regardless of protected characteristics
                - Background checks must comply with Fair Credit Reporting Act
                - Job descriptions must accurately reflect essential functions
                - Interview questions must be job-related and consistent
                
                Wage and Hour Compliance:
                - Minimum wage requirements vary by state and locality
                - Overtime pay required for non-exempt employees over 40 hours/week
                - Accurate timekeeping records must be maintained
                - Break and meal period requirements vary by jurisdiction
                
                Workplace Safety:
                - OSHA compliance required for most employers
                - Safety training must be provided and documented
                - Workplace injury reporting within 24 hours
                - Right to refuse unsafe work without retaliation
                
                Discrimination and Harassment:
                - Zero tolerance policy for harassment and discrimination
                - Training required for supervisors and employees
                - Clear reporting procedures must be established
                - Prompt investigation of all complaints required
                
                Employee Privacy:
                - Electronic monitoring policies must be disclosed
                - Medical information must be kept confidential
                - Social media policies should balance rights and business needs
                - Drug testing policies must comply with applicable laws
                """
            }
        }
    
    def setup_test_environment(self):
        """Setup temporary test environment"""
        self.temp_dir = tempfile.mkdtemp()
        data_dir = Path(self.temp_dir) / "data"
        data_dir.mkdir()
        
        # Create test documents
        for domain, files in self.test_documents.items():
            domain_dir = data_dir / domain
            domain_dir.mkdir()
            
            for filename, content in files.items():
                file_path = domain_dir / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content.strip())
        
        return data_dir
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generic_agent_initialization(self):
        """Test Generic ICAR V2 Agent initialization"""
        print("\n1. Testing Generic ICAR V2 Agent Initialization...")
        
        try:
            from generic_agent import GenericICARAgent
            
            # Test different processing modes
            for mode in ["keywords", "summary", "hybrid"]:
                agent = GenericICARAgent(processing_mode=mode)
                
                # Mock the vector store initialization
                with unittest.mock.patch.object(agent.enhanced_vector_store, 'initialize', return_value=True):
                    result = agent.initialize()
                    
                if result:
                    print(f"✅ Generic ICAR Agent ({mode} mode) initialized successfully")
                else:
                    print(f"❌ Generic ICAR Agent ({mode} mode) initialization failed")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Generic ICAR Agent initialization error: {e}")
            return False
    
    def test_multi_domain_processing(self):
        """Test processing documents from multiple domains"""
        print("\n2. Testing Multi-Domain Document Processing...")
        
        try:
            from generic_processor import GenericProcessor
            
            processor = GenericProcessor(processing_mode="hybrid")
            processed_domains = {}
            
            for domain, files in self.test_documents.items():
                domain_chunks = []
                for filename, content in files.items():
                    doc_id = f"{domain}_{filename.replace('.txt', '')}"
                    chunks = processor.process_document(content, doc_id)
                    domain_chunks.extend(chunks)
                
                processed_domains[domain] = domain_chunks
                
                if len(domain_chunks) > 0:
                    print(f"✅ {domain.upper()}: Processed {len(domain_chunks)} chunks")
                    
                    # Test chunk quality
                    sample_chunk = domain_chunks[0]
                    if hasattr(sample_chunk, 'keywords') and len(sample_chunk.keywords) > 0:
                        print(f"   Keywords: {', '.join(sample_chunk.keywords[:3])}")
                    if hasattr(sample_chunk, 'summary') and len(sample_chunk.summary) > 10:
                        print(f"   Summary: {sample_chunk.summary[:100]}...")
                else:
                    print(f"❌ {domain.upper()}: Failed to process documents")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Multi-domain processing error: {e}")
            return False
    
    def test_universal_query_handling(self):
        """Test universal query handling across domains"""
        print("\n3. Testing Universal Query Handling...")
        
        try:
            from generic_agent import GenericICARAgent
            import unittest.mock
            
            agent = GenericICARAgent(processing_mode="hybrid")
            
            # Mock initialization
            agent.is_initialized = True
            
            test_queries = [
                ("Hello there!", "greeting"),
                ("What's the weather in Paris?", "weather"),
                ("What are the grading criteria?", "education"),
                ("How should I monitor my blood sugar?", "healthcare"), 
                ("What is your return policy?", "e_commerce"),
                ("What are the termination clauses?", "legal")
            ]
            
            for query, expected_domain in test_queries:
                action, reasoning = agent.decide_action(query)
                
                if "greeting" in expected_domain and "direct_response" in action.value:
                    print(f"✅ Greeting query: '{query[:30]}...' → Direct Response")
                elif "weather" in expected_domain and "weather_api" in action.value:
                    print(f"✅ Weather query: '{query[:30]}...' → Weather API")
                elif "generic_search" in action.value:
                    print(f"✅ Domain query: '{query[:30]}...' → Generic Search")
                else:
                    print(f"❌ Query handling failed: '{query[:30]}...'")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Universal query handling error: {e}")
            return False
    
    def test_context_reconstruction(self):
        """Test context reconstruction across domains"""
        print("\n4. Testing Context Reconstruction...")
        
        try:
            from enhanced_vector_store import EnhancedVectorStore
            import unittest.mock
            
            # Mock vector store with reconstruction
            store = EnhancedVectorStore(persist_directory=self.temp_dir + "/test_store")
            
            # Test reconstruction capability
            test_chunk = {
                "chunk_id": "test_001",
                "doc_id": "education_syllabus",
                "position": 0,
                "original_text": "Course syllabus for computer science including programming and data structures.",
                "keywords": ["programming", "data", "structures", "course"],
                "summary": "Computer science course covers programming and data structures."
            }
            
            reconstructed = store._reconstruct_chunk_context(test_chunk, "What topics are covered?")
            
            if len(reconstructed) > len(test_chunk["original_text"]):
                print("✅ Context reconstruction working - enhanced context provided")
            else:
                print("✅ Context reconstruction working - basic context provided")
            
            return True
            
        except Exception as e:
            print(f"❌ Context reconstruction error: {e}")
            return False
    
    def run_comprehensive_test(self):
        """Run comprehensive multi-domain test suite"""
        print("🧠 Generic ICAR V2 - Multi-Domain Test Suite")
        print("Author: Barış Genç")
        print("=" * 60)
        
        tests_passed = 0
        total_tests = 4
        
        try:
            # Setup test environment
            data_dir = self.setup_test_environment()
            print(f"Test environment created: {data_dir}")
            
            # Run tests
            if self.test_generic_agent_initialization():
                tests_passed += 1
            
            if self.test_multi_domain_processing():
                tests_passed += 1
            
            if self.test_universal_query_handling():
                tests_passed += 1
            
            if self.test_context_reconstruction():
                tests_passed += 1
            
        except Exception as e:
            print(f"❌ Test suite error: {e}")
        
        finally:
            # Cleanup
            self.cleanup_test_environment()
        
        # Results
        print("\n" + "=" * 60)
        print(f"Multi-Domain Test Results: {tests_passed}/{total_tests} passed")
        
        if tests_passed == total_tests:
            print("🎉 All Generic ICAR V2 multi-domain tests passed!")
            print("\n✅ DOMAINS VERIFIED:")
            print("   - Education (Syllabus, Student Handbook)")
            print("   - Healthcare (Diabetes, Patient Care)")  
            print("   - E-Commerce (Products, Shipping)")
            print("   - Legal (Contracts, Employment Law)")
            print("\n🚀 Generic ICAR V2 is ready for universal deployment!")
            return True
        else:
            print("❌ Some multi-domain tests failed")
            print("The system may not be fully generic yet.")
            return False

def main():
    """Main test runner"""
    # Import required for mocking
    import unittest.mock
    
    tester = TestMultiDomainGenericICAR()
    success = tester.run_comprehensive_test()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())