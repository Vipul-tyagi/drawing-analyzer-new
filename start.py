#!/usr/bin/env python3
"""
Universal starter script that works with or without full Python setup
Includes fallbacks for environments with limited Python support
"""

import os
import sys
from pathlib import Path

def safe_import_with_fallback():
    """Safely import modules with fallbacks for limited environments"""
    modules = {}
    
    # Try importing webbrowser with fallback
    try:
        import webbrowser
        modules['webbrowser'] = webbrowser
    except (ImportError, ModuleNotFoundError):
        print("⚠️ webbrowser module not available - browser won't auto-open")
        modules['webbrowser'] = None
    
    # Try importing http.server with fallback
    try:
        import http.server
        import socketserver
        modules['http_server'] = http.server
        modules['socketserver'] = socketserver
    except (ImportError, ModuleNotFoundError):
        print("⚠️ HTTP server modules not available")
        modules['http_server'] = None
        modules['socketserver'] = None
    
    # Try importing threading
    try:
        import threading
        import time
        modules['threading'] = threading
        modules['time'] = time
    except (ImportError, ModuleNotFoundError):
        print("⚠️ Threading modules not available")
        modules['threading'] = None
        modules['time'] = None
    
    return modules

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

def start_simple_server(modules):
    """Start a simple HTTP server for the web interface"""
    if not modules['http_server'] or not modules['socketserver']:
        print("❌ Cannot start HTTP server - required modules not available")
        print("💡 Try running: python3 -m http.server 8000")
        return False
    
    PORT = 8000
    
    class Handler(modules['http_server'].SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=os.getcwd(), **kwargs)
    
    try:
        with modules['socketserver'].TCPServer(("", PORT), Handler) as httpd:
            print(f"🌐 Simple web server running at http://localhost:{PORT}")
            print("📝 This provides the basic web interface with demo functionality")
            print("💡 For full AI analysis, set up the Python components")
            
            # Open browser if available
            if modules['webbrowser']:
                modules['webbrowser'].open(f'http://localhost:{PORT}')
            else:
                print(f"🔗 Please open http://localhost:{PORT} in your browser")
            
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        return True
    except Exception as e:
        print(f"❌ Server error: {e}")
        return False

def start_streamlit():
    """Start Streamlit application"""
    try:
        import subprocess
        print("🚀 Starting Streamlit application...")
        result = subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Streamlit failed: {e}")
        return False

def start_flask():
    """Start Flask server"""
    try:
        print("🚀 Starting Flask server...")
        import subprocess
        result = subprocess.run([sys.executable, "server.py"], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Flask failed: {e}")
        return False

def try_alternative_methods():
    """Try alternative startup methods when main methods fail"""
    print("\n🔄 Trying alternative startup methods...")
    
    # Check if we can run basic Python HTTP server
    try:
        print("💡 Attempting to start basic HTTP server...")
        os.system(f"{sys.executable} -m http.server 8000")
        return True
    except Exception as e:
        print(f"❌ Basic HTTP server failed: {e}")
    
    # Check if index.html exists for direct opening
    if os.path.exists("index.html"):
        print("📄 Found index.html - you can open it directly in your browser")
        print(f"🔗 File path: {os.path.abspath('index.html')}")
        return True
    
    return False

def main():
    """Main starter function with enhanced error handling"""
    print("🎨 Children's Drawing Analysis System")
    print("=" * 60)
    
    # Safely import modules
    modules = safe_import_with_fallback()
    
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
    print("🌐 Attempting to start simple web server...")
    
    if start_simple_server(modules):
        return
    
    # If all else fails, try alternative methods
    if try_alternative_methods():
        return
    
    # Final fallback
    print("\n❌ All startup methods failed")
    print("🔧 Troubleshooting suggestions:")
    print("   1. Check your Python installation")
    print("   2. Try: python3 -m http.server 8000")
    print("   3. Open index.html directly in your browser")
    print("   4. Install missing dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        print("🔧 This may be due to a Python installation issue")
        print("💡 Try running the application components individually:")
        print("   - For web interface: python3 -m http.server 8000")
        print("   - For Streamlit: streamlit run app.py")
        print("   - For Flask: python3 server.py")