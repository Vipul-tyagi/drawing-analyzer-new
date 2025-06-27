#!/usr/bin/env python3
"""
Universal starter script that works with or without full Python setup
"""

import os
import sys
import webbrowser
import http.server
import socketserver
import threading
import time
from pathlib import Path

def check_python_components():
    """Check if Python analysis components are available"""
    try:
        # Try importing key components
        import streamlit
        return True, "streamlit"
    except ImportError:
        pass
    
    try:
        import flask
        return True, "flask"
    except ImportError:
        pass
    
    return False, "none"

def start_simple_server():
    """Start a simple HTTP server for the web interface"""
    PORT = 8000
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=os.getcwd(), **kwargs)
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"🌐 Simple web server running at http://localhost:{PORT}")
            print("📝 This provides the basic web interface with demo functionality")
            print("💡 For full AI analysis, set up the Python components")
            
            # Open browser
            webbrowser.open(f'http://localhost:{PORT}')
            
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")

def start_streamlit():
    """Start Streamlit application"""
    try:
        import subprocess
        print("🚀 Starting Streamlit application...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except Exception as e:
        print(f"❌ Streamlit failed: {e}")
        return False
    return True

def start_flask():
    """Start Flask server"""
    try:
        print("🚀 Starting Flask server...")
        os.system(f"{sys.executable} server.py")
    except Exception as e:
        print(f"❌ Flask failed: {e}")
        return False
    return True

def main():
    """Main starter function"""
    print("🎨 Children's Drawing Analysis System")
    print("=" * 60)
    
    # Check what's available
    has_python, framework = check_python_components()
    
    if has_python:
        print(f"✅ Python components available ({framework})")
        
        if framework == "streamlit":
            print("🚀 Starting with Streamlit (full features)...")
            if start_streamlit():
                return
        elif framework == "flask":
            print("🚀 Starting with Flask server (full features)...")
            if start_flask():
                return
    
    print("⚠️ Python components not available or failed")
    print("🌐 Starting simple web server (demo mode)...")
    start_simple_server()

if __name__ == "__main__":
    main()