#!/usr/bin/env python3
"""
Application runner with automatic setup and error handling
"""

import os
import sys
import subprocess
from pathlib import Path

def check_streamlit():
    """Check if Streamlit is available"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """Install Streamlit if not available"""
    print("📦 Installing Streamlit...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        return True
    except subprocess.CalledProcessError:
        return False

def run_setup():
    """Run the setup script"""
    print("🔧 Running setup...")
    try:
        subprocess.check_call([sys.executable, "setup_environment.py"])
        return True
    except subprocess.CalledProcessError:
        return False

def run_streamlit():
    """Run the Streamlit application"""
    print("🚀 Starting Streamlit application...")
    
    # Set environment variables for better performance
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
    os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Failed to run application: {e}")

def main():
    """Main runner function"""
    print("🎨 Children's Drawing Analysis System")
    print("=" * 50)
    
    # Check if app.py exists
    if not Path("app.py").exists():
        print("❌ app.py not found!")
        print("Make sure you're in the correct directory")
        return False
    
    # Check Streamlit
    if not check_streamlit():
        print("⚠️ Streamlit not found, installing...")
        if not install_streamlit():
            print("❌ Failed to install Streamlit")
            return False
    
    # Run setup if needed
    if Path("setup_environment.py").exists():
        if not run_setup():
            print("⚠️ Setup failed, but continuing...")
    
    # Run the application
    run_streamlit()
    
    return True

if __name__ == "__main__":
    main()