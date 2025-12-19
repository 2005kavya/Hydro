#!/usr/bin/env python3
"""
HydroAlert Setup Script
Automatically installs all required packages and sets up the environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_packages():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    # Upgrade pip first
    if not run_command("python -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install packages from requirements.txt
    if not run_command("pip install -r requirements.txt", "Installing packages from requirements.txt"):
        return False
    
    # Install additional packages that might be needed
    additional_packages = [
        "streamlit",
        "plotly", 
        "pandas",
        "fastapi",
        "uvicorn[standard]"
    ]
    
    for package in additional_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️ Warning: {package} installation had issues, but continuing...")
    
    return True

def create_directories():
    """Create necessary directories if they don't exist"""
    print("\n📁 Creating necessary directories...")
    
    directories = [
        "backend/data",
        "backend/logs",
        "frontend/uploads"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_dependencies():
    """Check if all dependencies are properly installed"""
    print("\n🔍 Checking dependencies...")
    
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found")
        return False
    
    try:
        import plotly
        print(f"✅ Plotly {plotly.__version__}")
    except ImportError:
        print("❌ Plotly not found")
        return False
    
    try:
        import fastapi
        print(f"✅ FastAPI {fastapi.__version__}")
    except ImportError:
        print("❌ FastAPI not found")
        return False
    
    try:
        import pandas
        print(f"✅ Pandas {pandas.__version__}")
    except ImportError:
        print("❌ Pandas not found")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚰 HydroAlert Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install packages
    if not install_packages():
        print("\n❌ Package installation failed. Please check the errors above.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please reinstall packages.")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Open a terminal and run: cd backend && python -m uvicorn main:app --reload --port 8000")
    print("2. Open another terminal and run: cd frontend && streamlit run streamlit_app.py")
    print("3. Open your browser to: http://localhost:8501")
    
    print("\n🚀 Happy hydrating!")

if __name__ == "__main__":
    main()
