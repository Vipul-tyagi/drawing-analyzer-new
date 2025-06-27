#!/usr/bin/env python3
"""
Main entry point for Children's Drawing Analysis System
Automatically detects and starts the best available interface
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check which dependencies are available"""
    available = {}
    
    # Check Streamlit
    try:
        import streamlit
        available['streamlit'] = True
    except ImportError:
        available['streamlit'] = False
    
    # Check Flask
    try:
        import flask
        available['flask'] = True
    except ImportError:
        available['flask'] = False
    
    # Check core analysis components
    try:
        import cv2
        import numpy as np
        import PIL
        available['core_analysis'] = True
    except ImportError:
        available['core_analysis'] = False
    
    # Check AI components
    try:
        import torch
        import transformers
        available['ai_components'] = True
    except ImportError:
        available['ai_components'] = False
    
    return available

def install_missing_dependencies():
    """Install missing dependencies"""
    print("📦 Installing missing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def start_streamlit():
    """Start Streamlit application"""
    print("🚀 Starting Streamlit application...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")

def start_flask():
    """Start Flask server"""
    print("🚀 Starting Flask server...")
    try:
        subprocess.run([sys.executable, "server.py"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

def start_simple_server():
    """Start simple HTTP server"""
    print("🌐 Starting simple web server...")
    try:
        subprocess.run([sys.executable, "-m", "http.server", "8000"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

def main():
    """Main function"""
    print("🎨 Children's Drawing Analysis System")
    print("=" * 60)
    
    # Check dependencies
    deps = check_dependencies()
    
    print("📊 System Status:")
    for component, available in deps.items():
        status = "✅" if available else "❌"
        print(f"  {status} {component.replace('_', ' ').title()}")
    
    # Determine best startup method
    if deps['streamlit'] and deps['core_analysis']:
        print("\n🎯 Starting with Streamlit (recommended)")
        start_streamlit()
    elif deps['flask'] and deps['core_analysis']:
        print("\n🎯 Starting with Flask server")
        start_flask()
    elif Path("index.html").exists():
        print("\n🎯 Starting with simple web interface")
        print("⚠️ Limited functionality - install Python dependencies for full features")
        start_simple_server()
    else:
        print("\n❌ Cannot start application")
        print("🔧 Please install dependencies:")
        print("   python -m pip install -r requirements.txt")
        
        if input("\n📦 Install dependencies now? (y/n): ").lower() == 'y':
            if install_missing_dependencies():
                main()  # Retry after installation

if __name__ == "__main__":
    main()