#!/usr/bin/env python3
"""
Test project structure and sample data without ML dependencies
"""

import os
import sys
from pathlib import Path

def test_files_exist():
    """Test if all critical files exist"""
    required_files = [
        'main.py', 'streamlit_app.py', 'requirements.txt', 
        'Dockerfile', 'docker-compose.yml', '.env.example',
        'src/agent_core.py', 'src/vector_store.py', 
        'src/document_processor.py', 'src/external_apis.py'
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        print(f"❌ Missing files: {missing}")
        return False
    else:
        print("✅ All critical files exist")
        return True

def test_sample_data():
    """Test sample data creation"""
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory missing")
        return False
    
    files = list(data_dir.glob("*"))
    if len(files) >= 6:  # 3 txt + 3 pdf files
        print(f"✅ Sample data created ({len(files)} files)")
        return True
    else:
        print(f"❌ Insufficient sample data ({len(files)} files)")
        return False

def test_code_structure():
    """Test if code files have basic structure"""
    try:
        # Check main.py has main function
        with open("main.py") as f:
            main_content = f.read()
            if "def main(" in main_content and "__main__" in main_content:
                print("✅ main.py structure correct")
            else:
                print("❌ main.py structure incorrect")
                return False
        
        # Check streamlit app structure
        with open("streamlit_app.py") as f:
            streamlit_content = f.read()
            if "streamlit" in streamlit_content and "def main(" in streamlit_content:
                print("✅ streamlit_app.py structure correct")
            else:
                print("❌ streamlit_app.py structure incorrect")
                return False
        
        # Check Docker files
        with open("Dockerfile") as f:
            dockerfile = f.read()
            if "FROM python" in dockerfile and "requirements.txt" in dockerfile:
                print("✅ Dockerfile structure correct")
            else:
                print("❌ Dockerfile structure incorrect")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Code structure test failed: {e}")
        return False

def test_requirements():
    """Test requirements.txt"""
    try:
        with open("requirements.txt") as f:
            requirements = f.read()
            critical_packages = ["langchain", "chromadb", "streamlit", "requests", "python-dotenv"]
            
            missing_packages = []
            for package in critical_packages:
                if package not in requirements.lower():
                    missing_packages.append(package)
            
            if missing_packages:
                print(f"❌ Missing packages in requirements.txt: {missing_packages}")
                return False
            else:
                print("✅ All critical packages in requirements.txt")
                return True
                
    except Exception as e:
        print(f"❌ Requirements test failed: {e}")
        return False

def main():
    """Run structure tests"""
    print("🏗️  ChatbotDemo Structure Test")
    print("=" * 35)
    
    tests = [
        ("File Existence", test_files_exist),
        ("Sample Data", test_sample_data),
        ("Code Structure", test_code_structure),
        ("Requirements", test_requirements)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
    
    print(f"\n" + "=" * 35)
    print(f"Structure Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 Project structure is complete!")
        print("\nNext steps:")
        print("1. Add API keys to .env file")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Run: python main.py")
        print("4. Or use Docker: docker-compose up")
        return True
    else:
        print("❌ Project structure incomplete")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)