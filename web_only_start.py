#!/usr/bin/env python3
"""
Web-only startup that completely avoids Python signal dependencies
Uses only basic HTTP server functionality
"""

import os
import sys
from pathlib import Path

def create_simple_server():
    """Create the simplest possible HTTP server"""
    server_code = '''
import http.server
import socketserver
import os

PORT = 8000

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.getcwd(), **kwargs)

print(f"Server running at http://localhost:{PORT}")
print("Open this URL in your browser")

try:
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        httpd.serve_forever()
except KeyboardInterrupt:
    print("\\nServer stopped")
'''
    
    with open('simple_server.py', 'w') as f:
        f.write(server_code)
    
    return 'simple_server.py'

def main():
    """Main web-only startup"""
    print("🌐 Web-Only Startup (No Signal Dependencies)")
    print("=" * 50)
    
    # Check if web files exist
    if not Path('index.html').exists():
        print("❌ index.html not found")
        print("💡 Make sure you have the web interface files")
        return False
    
    # Create simple server
    server_file = create_simple_server()
    print(f"✅ Created {server_file}")
    
    # Try to start server
    try:
        print("🚀 Starting web server...")
        os.system(f"{sys.executable} {server_file}")
    except Exception as e:
        print(f"❌ Server failed: {e}")
        print("💡 Try opening index.html directly in your browser")
        html_path = Path('index.html').absolute()
        print(f"🔗 File path: file://{html_path}")

if __name__ == "__main__":
    main()