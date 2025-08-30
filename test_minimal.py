#!/usr/bin/env python3
"""
Minimal test without heavy ML dependencies
"""

import os
import sys
from pathlib import Path

def test_basic_functionality():
    """Test core functionality without ML dependencies"""
    print("Testing basic imports...")
    
    try:
        # Test document processor
        sys.path.append('src')
        from document_processor import DocumentProcessor
        
        dp = DocumentProcessor()
        print("✅ Document processor created")
        
        # Test text file loading
        text = dp.load_text_file("data/company_handbook.txt")
        if text and len(text) > 100:
            print("✅ Text file loading works")
        else:
            print("❌ Text file loading failed")
            return False
        
        # Test PDF loading
        pdf_text = dp.load_pdf("data/company_handbook.pdf")
        if pdf_text and len(pdf_text) > 100:
            print("✅ PDF loading works")
        else:
            print("❌ PDF loading failed")
            return False
        
        # Test external API structure
        from external_apis import WeatherAPI
        api = WeatherAPI()
        print("✅ Weather API structure works")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_environment():
    """Test environment setup"""
    print("\nTesting environment...")
    
    # Test .env.example
    if Path(".env.example").exists():
        print("✅ .env.example exists")
        with open(".env.example") as f:
            content = f.read()
            if "OPENAI_API_KEY" in content and "OPENWEATHER_API_KEY" in content:
                print("✅ Required keys in .env.example")
                return True
            else:
                print("❌ Missing keys in .env.example")
                return False
    else:
        print("❌ .env.example missing")
        return False

def main():
    print("🤖 Minimal ChatbotDemo Test")
    print("=" * 30)
    
    tests_passed = 0
    
    if test_basic_functionality():
        tests_passed += 1
    
    if test_environment():
        tests_passed += 1
    
    print(f"\nResults: {tests_passed}/2 tests passed")
    
    if tests_passed == 2:
        print("✅ Core functionality verified!")
        print("\nProject ready for deployment with API keys.")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)