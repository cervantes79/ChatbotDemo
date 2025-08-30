#!/usr/bin/env python3
"""
Basic test script to verify the system components work without heavy dependencies
"""

import os
import sys
from pathlib import Path

def test_project_structure():
    """Test if all required files exist"""
    required_files = [
        'main.py',
        'streamlit_app.py',
        'src/agent_core.py',
        'src/vector_store.py',
        'src/document_processor.py',
        'src/external_apis.py',
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml',
        '.env.example',
        '.gitignore'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("✅ All required files exist")
        return True

def test_sample_data():
    """Test sample data creation"""
    try:
        print("Testing sample data creation...")
        import subprocess
        result = subprocess.run([sys.executable, "create_sample_pdfs.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Sample data creation script works")
            
            # Check if files were created
            data_dir = Path("data")
            if data_dir.exists():
                files = list(data_dir.glob("*"))
                print(f"   Created {len(files)} files in data/ directory")
                return True
        else:
            print(f"❌ Sample data creation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing sample data: {e}")
        return False

def test_imports():
    """Test if basic imports work"""
    try:
        print("Testing basic imports...")
        
        # Test document processor
        from src.document_processor import DocumentProcessor
        dp = DocumentProcessor()
        print("✅ Document processor import works")
        
        # Test external API (without actual API calls)
        from src.external_apis import WeatherAPI
        wa = WeatherAPI()
        print("✅ Weather API import works")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_environment_setup():
    """Test environment configuration"""
    print("Testing environment setup...")
    
    env_example = Path(".env.example")
    if not env_example.exists():
        print("❌ .env.example file missing")
        return False
    
    with open(env_example) as f:
        content = f.read()
        if "OPENAI_API_KEY" in content:
            print("✅ Environment template looks correct")
            return True
        else:
            print("❌ .env.example missing required keys")
            return False

def main():
    """Run all basic tests"""
    print("🤖 ChatbotDemo - Basic System Test")
    print("=" * 40)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Sample Data Creation", test_sample_data), 
        ("Basic Imports", test_imports),
        ("Environment Setup", test_environment_setup)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 40)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All basic tests passed! System ready for deployment.")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and add your API keys")
        print("2. Run 'python main.py' for CLI interface")
        print("3. Or run 'docker-compose up' for containerized deployment")
    else:
        print("❌ Some tests failed. Please fix the issues before deployment.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)