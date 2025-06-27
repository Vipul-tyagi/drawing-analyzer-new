#!/usr/bin/env python3
"""
Run the application without signal module dependencies
This is the main alternative startup script
"""

import os
import sys
from pathlib import Path

# Set environment variables to avoid signal issues
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'

def check_available_methods():
    """Check which startup methods are available"""
    methods = {}
    
    # Check if we can import basic modules
    try:
        import http.server
        import socketserver
        methods['http_server'] = True
    except ImportError:
        methods['http_server'] = False
    
    # Check if streamlit is available
    try:
        import streamlit
        methods['streamlit'] = True
    except ImportError:
        methods['streamlit'] = False
    
    # Check if web files exist
    methods['web_files'] = Path('index.html').exists() and Path('app.js').exists()
    
    # Check if minimal app exists
    methods['minimal_app'] = Path('minimal_app.py').exists()
    
    return methods

def start_http_server():
    """Start HTTP server without signal dependencies"""
    try:
        import http.server
        import socketserver
        
        PORT = 8000
        
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=os.getcwd(), **kwargs)
        
        print(f"🌐 HTTP Server running at http://localhost:{PORT}")
        print("📝 Open the URL in your browser for the web interface")
        
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            httpd.serve_forever()
            
    except Exception as e:
        print(f"❌ HTTP server failed: {e}")
        return False

def start_streamlit_safe():
    """Start Streamlit with safe parameters"""
    try:
        import subprocess
        
        # Use subprocess to avoid direct signal imports
        cmd = [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ]
        
        print("🚀 Starting Streamlit with safe parameters...")
        subprocess.run(cmd)
        
    except Exception as e:
        print(f"❌ Streamlit failed: {e}")
        return False

def start_minimal_app():
    """Start minimal app version"""
    try:
        import subprocess
        
        if Path('minimal_app.py').exists():
            print("🚀 Starting minimal app...")
            subprocess.run([sys.executable, "minimal_app.py"])
        else:
            print("❌ minimal_app.py not found")
            return False
            
    except Exception as e:
        print(f"❌ Minimal app failed: {e}")
        return False

def open_web_files():
    """Open web files directly"""
    html_file = Path('index.html').absolute()
    
    if html_file.exists():
        print("📄 Web interface available!")
        print(f"🔗 Open this in your browser: file://{html_file}")
        print("💡 Or use a local HTTP server for better functionality")
        return True
    else:
        print("❌ index.html not found")
        return False

def main():
    """Main function to run without signal dependencies"""
    print("🎨 Children's Drawing Analysis - Signal-Free Startup")
    print("=" * 60)
    
    # Check available methods
    methods = check_available_methods()
    
    print("📊 Available startup methods:")
    for method, available in methods.items():
        status = "✅" if available else "❌"
        print(f"  {status} {method.replace('_', ' ').title()}")
    
    # Try methods in order of preference
    if methods['streamlit']:
        print("\n🚀 Attempting Streamlit startup...")
        try:
            start_streamlit_safe()
        except KeyboardInterrupt:
            print("\n👋 Streamlit stopped")
        except Exception as e:
            print(f"❌ Streamlit failed: {e}")
            print("🔄 Trying alternative methods...")
    
    if methods['http_server'] and methods['web_files']:
        print("\n🌐 Starting HTTP server for web interface...")
        try:
            start_http_server()
        except KeyboardInterrupt:
            print("\n👋 HTTP server stopped")
        except Exception as e:
            print(f"❌ HTTP server failed: {e}")
    
    elif methods['minimal_app']:
        print("\n🔧 Starting minimal app...")
        start_minimal_app()
    
    elif methods['web_files']:
        print("\n📄 Using direct web files...")
        open_web_files()
    
    else:
        print("\n❌ No startup methods available")
        print("🔧 Try running the fix script first:")
        print("   python fix_signal_error.py")

if __name__ == "__main__":
    main()